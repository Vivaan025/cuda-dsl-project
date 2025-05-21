# runtime/runtime.py

def simulate_vector_add(a, b):
    # Simulate out = a + b * 2
    return [ai + bi * 2 for ai, bi in zip(a, b)]

def test_simulation():
    a = [1, 2, 3]
    b = [4, 5, 6]
    out = simulate_vector_add(a, b)
    assert out == [1+4*2, 2+5*2, 3+6*2]
    print("runtime simulation passed.")

if __name__ == "__main__":
    test_simulation()
