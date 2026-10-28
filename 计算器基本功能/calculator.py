def calculate(operation, a, b):
    if operation == "sum":
        return a + b

    if operation == "multiply":
        return a * b

    if operation == "subtract":
        return a - b

    if operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Data_B 不能为 0")
        return a / b

    raise ValueError("不支持的 Operation")