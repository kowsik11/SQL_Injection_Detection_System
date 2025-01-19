from braket.circuits import Circuit
from braket.devices import LocalSimulator
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

# Quantum Key Distribution (BB84 Simulation) using Amazon Braket
def simulate_qkd_bb84():
    # Initialize Amazon Braket's local simulator
    device = LocalSimulator()

    # Create a BB84 Circuit
    circuit = Circuit()
    
    # Step 1: Prepare qubits in superposition (Alice's random bit preparation)
    circuit.h(0)  # Hadamard gate applied to qubit 0
    circuit.h(1)  # Hadamard gate applied to qubit 1
    
    # Step 2: Alice encodes qubits in random bases (X or Z)
    circuit.rx(0, 1.57)  # Rotate qubit 0 in the X basis
    circuit.rx(1, 1.57)  # Rotate qubit 1 in the X basis
    
    # Step 3: Bob measures the qubits
    circuit.measure(0)  # Measure qubit 0
    circuit.measure(1)  # Measure qubit 1
    
    # Execute the Circuit
    result = device.run(circuit, shots=100).result()
    counts = result.measurement_counts
    print(f"QKD Simulation - Measurement Counts: {counts}")
    
    # Generate shared key from the measurements (mockup of actual QKD process)
    # Extract the first bit of measurements as a simulated shared key
    shared_key = "".join([key[0] for key in counts.keys()])[:32]  # 256-bit key (32 bytes)
    return shared_key

# AES-GCM Encryption
def aes_encrypt(key, plaintext):
    # Generate a random initialization vector (IV)
    iv = os.urandom(12)  # 96-bit IV
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext, iv, encryptor.tag

# AES-GCM Decryption
def aes_decrypt(key, ciphertext, iv, tag):
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext

# Main Function: Quantum-Safe Encryption Workflow
def main():
    # Step 1: Simulate QKD to generate a quantum key
    print("Starting Quantum Key Distribution (QKD)...")
    quantum_key = simulate_qkd_bb84()
    print(f"Quantum Key (Binary String): {quantum_key}")
    
    # Convert quantum key to a 256-bit key for AES encryption
    required_key_length = 256  # AES-256
    quantum_key_padded = quantum_key.ljust(required_key_length, '0')  # Pad with zeros if too short
    aes_key = bytes(int(quantum_key_padded[i:i+8], 2) for i in range(0, required_key_length, 8))
    print(f"AES Key (Derived from Quantum Key): {aes_key.hex()}")
    
    # Step 2: Encrypt data using the quantum-derived key
    plaintext = b"Quantum-safe encryption example with Amazon Braket!"
    print(f"Plaintext: {plaintext}")
    
    ciphertext, iv, tag = aes_encrypt(aes_key, plaintext)
    print(f"Ciphertext: {ciphertext.hex()}")
    print(f"IV: {iv.hex()}")
    print(f"Tag: {tag.hex()}")
    
    # Step 3: Decrypt the ciphertext to verify correctness
    decrypted_text = aes_decrypt(aes_key, ciphertext, iv, tag)
    print(f"Decrypted Text: {decrypted_text}")
    
    # Verify decryption matches original plaintext
    assert decrypted_text == plaintext, "Decryption failed: mismatch with original plaintext"
    print("Encryption and decryption were successful.")

# Run the main function
if __name__ == "__main__":
    main()
