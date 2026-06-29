"""
Utility for suppressing callback output while still executing it.

Use this in Jupyter notebooks when you want the callback to run
(so you see the agent's reasoning locally) but don't want the
expandable suboutputs saved to the .ipynb file.
"""

from io import StringIO
import sys


def run_with_suppressed_output(runner, prompt, callback):
    """
    Run runner.loop() with callback but suppress output from being saved.
    
    Args:
        runner: The ToyAIKit runner object
        prompt: The prompt string to send to the agent
        callback: The callback to run (output is suppressed from saving)
    
    Returns:
        result: The LoopResult object
    
    Example:
        result = run_with_suppressed_output(runner, "Your question here?", callback)
        
        if result.all_messages:
            for msg in reversed(result.all_messages):
                if hasattr(msg, 'text'):
                    print(msg.text)
                    break
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    result = runner.loop(
        prompt=prompt,
        callback=callback
    )
    
    sys.stdout = old_stdout
    return result
