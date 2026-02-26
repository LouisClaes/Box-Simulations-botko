import sys
import threading
import time

def hang_detector():
    time.sleep(10)
    print("FATAL: HANG DETECTED. Printing stack trace of main thread.", file=sys.stderr)
    import traceback
    for th in threading.enumerate():
        if th is threading.main_thread():
            traceback.print_stack(sys._current_frames()[th.ident])
    import os
    os._exit(1)

t = threading.Thread(target=hang_detector, daemon=True)
t.start()

print("Importing modules...")
from strategies.two_bounded_best_fit.strategy import TwoBoundedBestFitStrategy
print("Import successful!")
