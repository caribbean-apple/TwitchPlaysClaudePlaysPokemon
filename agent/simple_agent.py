import base64
import copy
import io
import logging
import os
import time
import threading, queue
from datetime import datetime
from pathlib import Path

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR

from agent.emulator import Emulator
from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add this to suppress Anthropic's HTTP request logs
logging.getLogger("anthropic").setLevel(logging.CRITICAL)
# And possibly also suppress any underlying HTTP library logs
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("http.client").setLevel(logging.CRITICAL)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay.

You should name yourself WACLAUD and your rival CLAUDE. You are the rival of another channel called ClaudePlaysPokemon, who chose Bulbasaur. When choosing your pokemon, choose in character.

When naming yourself and your rival, be very sure the cursor > is immediately to the left of the letter you want to choose."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "name": "press_button",
        "description": "Press a button on the Game Boy. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "enum": ["a", "b", "start", "up", "down", "left", "right"],
                    "description": "The button to press. Valid buttons: 'a', 'b', 'start', 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing the button. Defaults to true."
                }
            },
            "required": ["button"],
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append({
        "name": "navigate_to",
        "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "The row coordinate to navigate to (0-8)."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate to navigate to (0-9)."
                }
            },
            "required": ["row", "col"],
        },
    })


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None, 
                 save_state_dir="saved_states", save_state_interval=10, auto_load_latest=True):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            load_state: Path to a saved state to load
            save_state_dir: Directory to save states to
            save_state_interval: Number of steps between state saves
            auto_load_latest: Whether to automatically load the latest save state
        """
        self.q = queue.Queue()
        self.emulator = Emulator(rom_path, self.q, headless=headless, sound=sound)
        self.emulator.initialize()  # Initialize the emulator
        self.client = Anthropic()
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        
        # State saving settings
        self.save_state_dir = save_state_dir
        self.save_state_interval = save_state_interval
        self.steps_since_last_save = 0
        
        # Create save state directory if it doesn't exist
        Path(self.save_state_dir).mkdir(parents=True, exist_ok=True)
        
        # Load state management
        if load_state:
            # If a specific state is provided, use that
            logger.info(f"Loading specified save state from {load_state}")
            self.emulator.load_state(load_state)
        elif auto_load_latest:
            # Try to load the latest save state
            latest_state = self.get_latest_save_state()
            if latest_state:
                logger.info(f"Auto-loading most recent save state: {latest_state}")
                self.emulator.load_state(latest_state)
                
    def get_latest_save_state(self):
        """Get the path to the most recent save state file, if any exist."""
        save_files = list(Path(self.save_state_dir).glob("*.state"))
        if not save_files:
            logger.info("No save states found to auto-load")
            return None
            
        # Sort by modification time (newest first)
        latest_save = max(save_files, key=lambda p: p.stat().st_mtime)
        return str(latest_save)

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        # logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_button":
            button = tool_input["button"]
            wait = tool_input.get("wait", True)
            print(f"Pressing button: {button}.\n")
            
            # Press button
            result = self.emulator.press_buttons(button, wait)
            # result = self.emulator.press_buttons([button], wait)
            
            # Get a fresh screenshot after executing the button
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Pressed button: {button}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after your button press:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            print(f"[Navigation] Navigating to: ({row}, {col})")
            # logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            
            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get a fresh screenshot after executing the navigation
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            # logger.info(f"[Memory State after action]")
            # logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            # if collision_map:
                # logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Navigation result: {result}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def save_current_state(self):
        """Save the current state of the emulator."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create a filename with timestamp only
        filename = f"{timestamp}.state"
        filepath = os.path.join(self.save_state_dir, filename)
        
        # Save the state
        self.emulator.save_state(filepath)
        logger.info(f"Saved game state to {filepath}")
                
        return filepath

    def run(self, num_steps=float('inf')):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for. By default, don't stop running.
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0

        while self.running:
            try:
                messages = copy.deepcopy(self.message_history)

                if len(messages) >= 3:
                    if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                    
                    if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                        messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}


                # Get model response
                response = self.client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    temperature=TEMPERATURE,
                )

                # logger.info(f"Response usage: {response.usage}")

                # Extract tool calls
                tool_calls = [
                    block for block in response.content if block.type == "tool_use"
                ]

                # Display the model's reasoning
                for block in response.content:
                    if block.type == "text":
                        print(f"<thinking> {block.text} </thinking>")
                        # logger.info(f"<thinking> {block.text} </thinking>")
                    # elif block.type == "tool_use":
                        # logger.info(f"[Tool] Using tool: {block.name}")

                # Process tool calls
                if tool_calls:
                    # Add assistant message to history
                    assistant_content = []
                    for block in response.content:
                        if block.type == "text":
                            assistant_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            assistant_content.append({"type": "tool_use", **dict(block)})
                    
                    self.message_history.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    
                    # Process tool calls and create tool results
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_result = self.process_tool_call(tool_call)
                        tool_results.append(tool_result)
                    
                    # Add tool results to message history
                    self.message_history.append(
                        {"role": "user", "content": tool_results}
                    )

                    # Check if we need to summarize the history
                    if len(self.message_history) >= self.max_history:
                        self.summarize_history()

                steps_completed += 1
                if (steps_completed%10 == 0): 
                    print(f"Reached {steps_completed} steps completed.")
                # Increment steps counter and save state if needed
                self.steps_since_last_save += 1
                if self.steps_since_last_save >= self.save_state_interval:
                    self.save_current_state()
                    self.steps_since_last_save = 0
                    
                # logger.info(f"Completed step {steps_completed}/{num_steps}")
                
                # Check if we've completed the requested number of steps
                if steps_completed >= num_steps and num_steps > 0:
                    break

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                # Save state on keyboard interrupt
                self.save_current_state()
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                # Try to save state on error
                try:
                    self.save_current_state()
                except:
                    pass
                raise e

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        # logger.info(f"[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Create messages for the summarization request - pass the entire conversation history
        messages = copy.deepcopy(self.message_history) 


        if len(messages) >= 3:
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            
            if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SUMMARY_PROMPT,
                    }
                ],
            }
        ]
        
        # Get summary from Claude
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        # Extract the summary text
        summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        
        # logger.info(f"[Agent] Message history condensed into summary.")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run()
        # logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()