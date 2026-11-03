"""
Student Grade Manager
Week 1 · Day 2 assignment

Run with:
    python grade_manager.py
"""

students = []


def get_grade(mark):
    """Return the grade for a mark from 0 to 100."""
    if mark >= 90:
        return "A"
    elif mark >= 80:
        return "B"
    elif mark >= 70:
        return "C"
    elif mark >= 60:
        return "D"
    return "E"


def add_student():
    """Ask for a student's name and mark, validate them, and store the record."""
    name = input("Enter student name: ").strip()

    if not name:
        print("Name cannot be empty. Student was not added.")
        return

    raw_mark = input("Enter mark (0-100): ").strip()

    try:
        mark = float(raw_mark)
    except ValueError:
        print("Invalid mark. Please enter a number (for example, 85 or 72.5).")
        return

    if not 0 <= mark <= 100:
        print("Mark must be between 0 and 100. Student was not added.")
        return

    # A list of dictionaries allows students with the same name to be stored.
    students.append({
        "name": name,
        "mark": mark,
        "grade": get_grade(mark),
    })
    print(f"{name} was added successfully.")


def show_results():
    """Print all student results and class statistics."""
    if not students:
        print("\nNo student records yet. Add a student first.")
        return

    name_width = max(4, max(len(student["name"]) for student in students))
    name_width = min(name_width, 30)

    print()
    print(f'{"Name":<{name_width}}  {"Mark":>7}  {"Grade":>5}')
    print("-" * (name_width + 18))

    for student in students:
        display_name = student["name"][:name_width]
        print(
            f'{display_name:<{name_width}}  '
            f'{student["mark"]:>7.1f}  '
            f'{student["grade"]:>5}'
        )

    marks = [student["mark"] for student in students]
    average = sum(marks) / len(marks)

    print()
    print(
        f"Average: {average:.1f}    "
        f"Highest: {max(marks):.1f}    "
        f"Lowest: {min(marks):.1f}"
    )


def main():
    """Run the menu until the user chooses to quit."""
    while True:
        print("\n=== Student Grade Manager ===")
        print("1. Add student")
        print("2. Show results")
        print("3. Quit")

        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            add_student()
        elif choice == "2":
            show_results()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
