"""Independent FIPS 203 K-PKE oracle: interpolation and schoolbook ring math.

No shader butterfly, twiddle table, or quadratic base-multiply code is reused.
Test laboratory only; not an OA implementation or secret execution path.
"""
import hashlib

Q = 3329
ROOTS = [pow(17, 2 * int(f"{i:07b}"[::-1], 2) + 1, Q) for i in range(128)]


def transform(poly):
    result = []
    for root in ROOTS:
        for parity in range(2):
            value = 0
            for coefficient in reversed(poly[parity::2]):
                value = (value * root + coefficient) % Q
            result.append(value)
    return result


def interpolate(values):
    result = [0] * 256
    for i, root in enumerate(ROOTS):
        inverse, power = pow(root, -1, Q), 1
        for degree in range(128):
            for parity in range(2):
                result[2 * degree + parity] += values[2 * i + parity] * power
            power = power * inverse % Q
    return [x * pow(128, -1, Q) % Q for x in result]


def product(a, b):
    result = [0] * 256
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[(i + j) % 256] += x * y * (1 if i + j < 256 else -1)
    return [x % Q for x in result]


def add(a, b):
    return [(x + y) % Q for x, y in zip(a, b)]


def matrix_poly(rho, i, j):
    stream = hashlib.shake_128(rho + bytes([j, i])).digest(4096)
    values = []
    for offset in range(0, len(stream) - 2, 3):
        a, b, c = stream[offset:offset + 3]
        values.extend(x for x in (a + 256 * (b % 16), b // 16 + 16 * c) if x < Q)
        if len(values) >= 256:
            return interpolate(values[:256])
    raise AssertionError("oracle XOF fixture capacity exhausted")


def noise(seed, nonce, eta):
    value = int.from_bytes(hashlib.shake_256(seed + bytes([nonce])).digest(64 * eta), "little")
    mask = (1 << eta) - 1
    result = []
    for _ in range(256):
        result.append(((value & mask).bit_count() - ((value >> eta) & mask).bit_count()) % Q)
        value >>= 2 * eta
    return result


def encode(poly, bits):
    return sum(x << (i * bits) for i, x in enumerate(poly)).to_bytes(32 * bits, "little")


def decode(data, bits):
    value = int.from_bytes(data, "little")
    return [((value >> (i * bits)) & ((1 << bits) - 1)) % Q for i in range(256)]


def compress(x, bits):
    quotient, remainder = divmod(x * (1 << bits), Q)
    return (quotient + (2 * remainder >= Q)) % (1 << bits)


def decompress(x, bits):
    quotient, remainder = divmod(x * Q, 1 << bits)
    return quotient + (2 * remainder >= 1 << bits)


def kpke(d, r, message, k):
    eta, du, dv = (3 if k == 2 else 2), (11 if k == 4 else 10), (5 if k == 4 else 4)
    expanded = hashlib.sha3_512(d + bytes([k])).digest()
    rho, sigma = expanded[:32], expanded[32:]
    a = [[matrix_poly(rho, i, j) for j in range(k)] for i in range(k)]
    s = [noise(sigma, i, eta) for i in range(k)]
    t = []
    for i in range(k):
        row = noise(sigma, k + i, eta)
        for j in range(k):
            row = add(row, product(a[i][j], s[j]))
        t.append(row)
    ek = b"".join(encode(transform(x), 12) for x in t) + rho
    dk = b"".join(encode(transform(x), 12) for x in s)
    y = [noise(r, i, eta) for i in range(k)]
    u = []
    for i in range(k):
        row = noise(r, k + i, 2)
        for j in range(k):
            row = add(row, product(a[j][i], y[j]))
        u.append(row)
    mu = [decompress(x, 1) for x in decode(message, 1)]
    v = add(noise(r, 2 * k, 2), mu)
    for i in range(k):
        v = add(v, product(t[i], y[i]))
    ciphertext = b"".join(encode([compress(x, du) for x in row], du) for row in u)
    ciphertext += encode([compress(x, dv) for x in v], dv)
    # Decode the serialized ciphertext independently, including quantization.
    v_prime = [decompress(x, dv) for x in decode(ciphertext[32 * du * k:], dv)]
    for i in range(k):
        u_prime = [decompress(x, du) for x in decode(ciphertext[i * 32 * du:(i + 1) * 32 * du], du)]
        term = product(s[i], u_prime)
        v_prime = [(x - y) % Q for x, y in zip(v_prime, term)]
    recovered = encode([compress(x, 1) for x in v_prime], 1)
    return ek, dk, ciphertext, recovered


def decrypt(dk, ciphertext, k):
    du, dv = (11 if k == 4 else 10), (5 if k == 4 else 4)
    value = [decompress(x, dv) for x in decode(ciphertext[32 * du * k:], dv)]
    for i in range(k):
        secret = interpolate(decode(dk[384 * i:384 * (i + 1)], 12))
        u = [decompress(x, du) for x in decode(ciphertext[i * 32 * du:(i + 1) * 32 * du], du)]
        term = product(secret, u)
        value = [(x - y) % Q for x, y in zip(value, term)]
    return encode([compress(x, 1) for x in value], 1)


def kem(d, z, message, k):
    # Existing independent ring oracle supplies K-PKE, hashlib supplies H/G/J.
    ek, secret, _, _ = kpke(d, bytes(32), bytes(32), k)
    h = hashlib.sha3_256(ek).digest()
    expanded = hashlib.sha3_512(message + h).digest()
    _, _, ciphertext, _ = kpke(d, expanded[32:], message, k)
    return ek, secret + ek + h + z, ciphertext, expanded[:32]


def encaps(ek, message, k):
    eta, du, dv = (3 if k == 2 else 2), (11 if k == 4 else 10), (5 if k == 4 else 4)
    expanded = hashlib.sha3_512(message + hashlib.sha3_256(ek).digest()).digest()
    y = [noise(expanded[32:], i, eta) for i in range(k)]
    u = []
    for i in range(k):
        row = noise(expanded[32:], k + i, 2)
        for j in range(k):
            row = add(row, product(matrix_poly(ek[-32:], j, i), y[j]))
        u.append(row)
    v = add(noise(expanded[32:], 2 * k, 2), [decompress(x, 1) for x in decode(message, 1)])
    for i in range(k):
        t = interpolate(decode(ek[384 * i:384 * (i + 1)], 12))
        v = add(v, product(t, y[i]))
    c = b"".join(encode([compress(x, du) for x in row], du) for row in u)
    return expanded[:32], c + encode([compress(x, dv) for x in v], dv)


def decaps(dk, ciphertext, k):
    message = decrypt(dk[:384 * k], ciphertext, k)
    ek = dk[384 * k:768 * k + 32]
    h = dk[768 * k + 32:768 * k + 64]
    expanded = hashlib.sha3_512(message + h).digest()
    # Re-encryption uses stored h for r, not a freshly computed H(ek).
    # Use independent coefficient-ring arithmetic with the derived r.
    # Valid laboratory keys have h=H(ek); malformed key admission is separate.
    if h != hashlib.sha3_256(ek).digest():
        raise AssertionError("decaps oracle expects a validated key")
    _, check = encaps(ek, message, k)
    return expanded[:32] if ciphertext == check else hashlib.shake_256(dk[-32:] + ciphertext).digest(32)
