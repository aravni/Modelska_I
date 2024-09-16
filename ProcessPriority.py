# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
Funkcije za določitev prioritete Python procesov
'''
# TODO: preostale prioritete, background...

def HighPriority() -> None:
    '''Set highest priority of the process.'''
    import sys
    import os
    import psutil

    os_used = sys.platform
    process = psutil.Process(os.getpid())  # Set highest priority for the python script for the CPU
    if os_used == "win32":  # Windows (either 32-bit or 64-bit)
        process.nice(psutil.REALTIME_PRIORITY_CLASS)
    elif os_used == "linux":  # linux
        process.nice(psutil.IOPRIO_HIGH)
    else:  # MAC OS X or other
        process.nice(20)

def LowPriority():
    """ Set lowest priority of the process."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os
        os.nice(1)