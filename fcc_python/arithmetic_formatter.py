def arithmetic_arranger(problems, show_answers=False):
    """
    Arrange arithmetic problems vertically and optionally show the answers.

    Args:
        problems (list): A list of arithmetic problems in the format "operand1 operator operand2".
                         The operator must be either '+' or '-', and the operands must be positive integers
                         with at most four digits.
        show_answers (bool, optional): If set to True, the function will include the answers in the output.

    Returns:
        str: The arranged arithmetic problems as a string. If `show_answers` is True, the answers are included
             in the output. If any errors are encountered, an error message is returned instead.
    """
    if len(problems) > 5:
        return 'Error: Too many problems.'

    first_line = ''
    second_line = ''
    dashes = ''
    answers = ''

    for problem in problems:
        first_operand, operator, second_operand = problem.split()

        if operator not in ['+', '-']:
            return "Error: Operator must be '+' or '-'."

        if not first_operand.isdigit() or not second_operand.isdigit():
            return 'Error: Numbers must only contain digits.'

        if len(first_operand) > 4 or len(second_operand) > 4:
            return 'Error: Numbers cannot be more than four digits.'

        width = max(len(first_operand), len(second_operand)) + 2

        first_line += f"{first_operand.rjust(width)}    "
        second_line += f"{operator}{second_operand.rjust(width - 1)}    "
        dashes += f"{'-' * width}    "

        if show_answers:
            if operator == '+':
                answer = str(int(first_operand) + int(second_operand))
            else:
                answer = str(int(first_operand) - int(second_operand))
            answers += f"{answer.rjust(width)}    "

    arranged_problems = f"{first_line.rstrip()}\n{second_line.rstrip()}\n{dashes.rstrip()}"

    if show_answers:
        arranged_problems += f"\n{answers.rstrip()}"

    return arranged_problems

# Example usage:
print(arithmetic_arranger(["32 + 698", "3801 - 2", "45 + 43", "123 + 49"]))
