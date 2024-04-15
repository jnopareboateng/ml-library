def arithmetic_arranger(problems, show_answers=False):
    if len(problems) > 5:
        return 'Error: Too many problems.'
    first_line = ''
    second_line = ''
    dashes = ''
    answers = ''
    arranged_problems = ''
    for problem in problems:
        first_operand, operator, second_operand = problem.split()
        if operator not in ['+', '-']:
            return "Error: Operator must be '+' or '-'."
        if not first_operand.isdigit() or not second_operand.isdigit():
            return 'Error: Numbers must only contain digits.'
        if len(first_operand) > 4 or len(second_operand) > 4:
            return 'Error: Numbers cannot be more than four digits.'
        width = max(len(first_operand), len(second_operand)) + 2
        first_line += str(first_operand).rjust(width) + '    '
        second_line += operator + str(second_operand).rjust(width - 1) + '    '
        dashes += '-' * width + '    '
        if show_answers:
            if operator == '+':
                answer = str(int(first_operand) + int(second_operand))
            else:
                answer = str(int(first_operand) - int(second_operand))
            answers += answer.rjust(width) + '    '
    arranged_problems += first_line.rstrip() + '\n' + second_line.rstrip() + '\n' + dashes.rstrip()
    if show_answers:
        arranged_problems += '\n' + answers.rstrip()
    return arranged_problems

# Example usage:
print(arithmetic_arranger(["32 + 698", "3801 - 2", "45 + 43", "123 + 49"]))
