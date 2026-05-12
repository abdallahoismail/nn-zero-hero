import time
import sys
import random
import itertools
import subprocess
import datetime as dt
# from PIL import Image


# ----------- CONFIG (customize this) -----------
HER_NAME = "Fatima, aka small pookie"
DAYS_TOGETHER = (dt.datetime.now() - dt.datetime(2023, 12, 5)).days
MEMORY_FILE = "our_memories_in_puerto_rico_and_miami.csv"

# Add your image paths here
IMAGE_1 = "my_photo.jpeg"  # e.g., you two together
IMAGE_2 = "my_photo2.jpeg"  # e.g., her smiling
IMAGE_3 = "my_photo3.jpeg"  # e.g., a memorable moment

FEATURES = [
    "your smile",
    "your jokes",
    "you lingo that you make upyour beautiful face",
    "your laugh",
    "your kindness",
    "your kisses",
]

FINAL_MESSAGE = f"""
Conclusion:
No matter how many iterations I run,
the result is always the same:

I choose you.
I love you, {HER_NAME}.
Happy Anniversary ❤️
"""
# ----------------------------------------------


# ----------- UTILITIES -----------
def slow_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def wait(min_t=0.3, max_t=0.8):
    time.sleep(random.uniform(min_t, max_t))


# ----------- SPINNER -----------
def spinner(task_name, duration=2):
    spinner_chars = itertools.cycle(["|", "/", "-", "\\"])
    end_time = time.time() + duration

    sys.stdout.write(f"{task_name} ")
    sys.stdout.flush()

    while time.time() < end_time:
        sys.stdout.write(next(spinner_chars))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write("\b")

    sys.stdout.write("✓\n")
    sys.stdout.flush()


# ----------- IMAGE DISPLAY (iTerm2) -----------
def show_image(path):
    try:
        subprocess.run(["imgcat", path])
    except FileNotFoundError:
        slow_print("[Image could not be displayed — imgcat not installed]")


# ----------- TRAINING LOOP -----------
def training_loop():
    loss = 1.0
    epochs = [1, 3, 5, 8, 12, 20, 35, 50]

    for epoch in epochs:
        wait()

        loss *= random.uniform(0.5, 0.8)

        slow_print(f"Epoch {epoch}/∞")
        slow_print(f"Loss: {loss:.4f} ❤️")

        if random.random() > 0.5:
            feature = random.choice(FEATURES)
            slow_print(f"Discovered feature: {feature}")

            # Show image when discovering something meaningful
            if "smile" in feature.lower():
                show_image(IMAGE_2)

        if epoch == 20:
            slow_print("Warning: Unable to minimize love below threshold")
            slow_print("Reason: feelings too strong ❤️")

        if epoch == 35:
            slow_print("Overfitting on: your laugh (acceptable)")

        print()


# ----------- MAIN -----------
def main():
    slow_print("Initializing model...")
    spinner("Building architecture", 2)

    slow_print(f"Model: LoveNet_v1 (trained on {HER_NAME})")

    spinner("Loading dataset", 2)
    slow_print(f"Dataset: {MEMORY_FILE}")

    # 🔥 Show first image during dataset loading
    show_image(IMAGE_1)

    slow_print(f"Samples loaded: {DAYS_TOGETHER} days")

    spinner("Compiling feelings", 2)

    slow_print("Defining objective function...")
    slow_print(f"Goal: maximize happiness({HER_NAME}) ❤️")

    print()
    slow_print("Starting training...\n")

    training_loop()

    slow_print("Training complete.")
    slow_print("Model converged.\n")

    time.sleep(1.5)

    slow_print("Final prediction:")
    slow_print("P(love_you_forever) = 1.0\n")

    # 🔥 Final emotional hit — show image again
    show_image(IMAGE_2)
    show_image(IMAGE_3)

    slow_print(FINAL_MESSAGE, delay=0.04)


if __name__ == "__main__":
    main()
