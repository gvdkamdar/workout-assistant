import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from db_utils import init_db, get_logs, add_dummy_data  # Database utilities
from exercise_counter import perform_exercise  # Exercise counting logic

# please run the file with 'cli' at the end in the following way
# main.py cli

# app = Flask(__name__)
# # Serve the web interface
# @app.route('/')
# def index():
#     return render_template('rep_counter_interface.html')

# # Handle exercise commands from the web interface
# @app.route('/start-exercise', methods=['POST'])
# def start_exercise():
#     data = request.json
#     exercise = data.get("exercise")
#     if exercise == "bicep_curl":
#         perform_exercise(1)
#     elif exercise == "bench_press":
#         perform_exercise(2)
#     elif exercise == "lateral_raise":
#         perform_exercise(3)
#     elif exercise == "shoulder_press":
#         perform_exercise(4)
#     else:
#         return jsonify({"error": "Unknown exercise"}), 400
#     return jsonify({"message": f"Started {exercise}"}), 200

# # Fetch workout logs for the web interface
# @app.route('/logs', methods=['GET'])
# def fetch_logs():
#     logs = get_logs()
#     if logs:
#         return jsonify(logs)
#     return jsonify({"message": "No workout logs found."}), 200

# Speech-to-Text Function for CLI
def recognize_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening for your command...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that. Please try again.")
        except sr.RequestError:
            print("There was an error with the speech recognition service.")
        except sr.WaitTimeoutError:
            print("No command detected. Please try again.")
    return None

# Main Speech-Based CLI Function
def main_cli():
    add_dummy_data()
    while True:
        print("\nAvailable Commands: bicep curls, bench press, lateral raises, shoulder presses, view log, exit.")
        command = recognize_command()

        if not command:
            continue  # Skip if no command was recognized

        if "exit" in command:
            print("Exiting the program. Goodbye!")
            break
        elif "view log" in command or "logs" in command:
            print("\nFetching workout logs...\n")
            logs = get_logs()
            if logs:
                print("Workout Logs:")
                for log in logs:
                    print(
                        f"User: {log[1]}, Exercise: {log[2]}, Reps: {log[3]}, "
                        f"Time per Rep: {log[4]:.2f}s, Total Time: {log[5]:.2f}s, Date: {log[6]}"
                    )
            else:
                print("No workout logs found.")
        elif "bicep curls" in command:
            perform_exercise(1)
        elif "bench press" in command:
            perform_exercise(2)
        elif "lateral raises" in command:
            perform_exercise(3)
        elif "shoulder presses" in command or "shoulder press" in command:
            perform_exercise(4)
        else:
            print("Invalid command. Please try again.")

if __name__ == "__main__":
    init_db()  # Ensure database is initialized

    # Choose between CLI and Web Interface
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        main_cli()  # Run the CLI interface
    else:
        pass
        # app.run(debug=True)  # Run the Flask web interface
