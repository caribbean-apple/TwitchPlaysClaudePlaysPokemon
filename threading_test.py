from pyboy import PyBoy
import threading, queue
import os
import random
import time

q = queue.Queue()
# current_dir = os.path.dirname(os.path.abspath(__file__))
rom_path = os.path.join(os.getcwd(), "pokemon.gb")

possible_buttons = ['a', 'b', 'left', 'right', 'up', 'down', 'start']

def player():
    try:
        print(f"Looking for ROM at: {rom_path}")
        print(f"ROM file exists: {os.path.exists(rom_path)}")

        pyboy = PyBoy(
            rom_path,
            cgb=True,
            sound=True,
        )
        print("PyBoy initialized successfully")
        
        while True:
            try:
                item = q.get(block=False)
                print(f"Processing button: {item}")
                pyboy.button(item)
                # pyboy.send_input(item)
            except queue.Empty:
                pass

            # tick() returns True to continue, False to stop
            if not pyboy.tick():  # Only exit when tick() returns False
                print("PyBoy tick returned False, exiting player thread")
                break
                
    except Exception as e:
        print(f"Error in player thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Player thread finished")

def comms():
    n_iteration = 0
    while True:
        n_iteration +=1
        # Print current time in a readable format
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Processing at: {current_time}")

        # busywork
        if n_iteration % 5 == 0:
            print("Taking the sum")
            maxnum = random.randint(10000000, 20000000)
            output = sum([k**2 for k in range(maxnum)])
            index = output % len(possible_buttons)
            item = possible_buttons[index]
        else:
            time.sleep(2)
            item = possible_buttons[random.randint(0, len(possible_buttons) - 1)]
        q.put(item)

t1 = threading.Thread(target=player)
t2 = threading.Thread(target=comms, daemon=True)

t1.start()
t2.start()

t1.join()
# t2.kill()  # This method doesn't exist
# Daemon threads automatically terminate when the main thread exits