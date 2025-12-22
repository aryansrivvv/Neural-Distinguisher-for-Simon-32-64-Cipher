import numpy as np
import os
from Simon import Simon3264

class SimonDataGenerator:
    def __init__(self):
        self.cipher = Simon3264()
        # Differential for Simon 32/64 (12 Rounds) from Table 3
        # Input Diff: (0x0, 0x40) -> High Word (L) 0x0, Low Word (R) 0x40
        self.diff_in_L = 0x0000
        self.diff_in_R = 0x0040

        # Key Diff: (0x0, 0x0, 0x0, 0x40) -> Only the last key word (k0) has difference
        # Recall key is (k3, k2, k1, k0). The paper lists (0x0, 0x0, 0x0, 0x40).
        self.diff_key = [0x0000, 0x0000, 0x0000, 0x0040]

    def generate_batch(self, batch_size=10000, rounds=12):
        """
        Generates a balanced batch of positive and negative samples.
        Returns:
            X: Ciphertext pairs (converted to binary/features)
            Y: Labels (1 for Related-Key, 0 for Random)
        """
        half_batch = batch_size // 2

        # --- 1. Generate Positive Samples (Label 1) ---
        # Random master keys K
        keys_pos = np.frombuffer(os.urandom(2 * 4 * half_batch), dtype=np.uint16).reshape(half_batch, 4)

        # Calculate related keys K' = K ^ diff_key
        # We need to broadcast the key difference
        diff_k_array = np.array(self.diff_key, dtype=np.uint16)
        keys_pos_prime = keys_pos ^ diff_k_array

        # Random plaintexts P (Left, Right)
        pt_pos = np.frombuffer(os.urandom(2 * 2 * half_batch), dtype=np.uint16).reshape(half_batch, 2)

        # Calculate related plaintexts P' = P ^ diff_in
        # diff_in is [Right, Left] because our encrypt takes [R, L] format in the previous code?
        # Let's verify standard: usually (Left, Right).
        # Paper says delta_p = (0x0, 0x40). Usually denotes (Left_Diff, Right_Diff).
        # My Simon3264.encrypt expects [Right, Left] order in the numpy array based on previous code logic.
        # Let's assume standard (L, R) for input params and convert.

        diff_p_array = np.array([self.diff_in_R, self.diff_in_L], dtype=np.uint16)
        pt_pos_prime = pt_pos ^ diff_p_array

        # Expand keys
        round_keys_pos = self.cipher.expand_key(keys_pos)
        round_keys_pos_prime = self.cipher.expand_key(keys_pos_prime)

        # Encrypt
        ct_pos = self.cipher.encrypt(pt_pos, round_keys_pos, rounds=rounds)
        ct_pos_prime = self.cipher.encrypt(pt_pos_prime, round_keys_pos_prime, rounds=rounds)

        # --- 2. Generate Negative Samples (Label 0) ---
        # Random keys K and K' (Independent)
        keys_neg = np.frombuffer(os.urandom(2 * 4 * half_batch), dtype=np.uint16).reshape(half_batch, 4)
        keys_neg_prime = np.frombuffer(os.urandom(2 * 4 * half_batch), dtype=np.uint16).reshape(half_batch, 4)

        # Random plaintexts P and P' (Independent)
        pt_neg = np.frombuffer(os.urandom(2 * 2 * half_batch), dtype=np.uint16).reshape(half_batch, 2)
        pt_neg_prime = np.frombuffer(os.urandom(2 * 2 * half_batch), dtype=np.uint16).reshape(half_batch, 2)

        # Expand keys
        round_keys_neg = self.cipher.expand_key(keys_neg)
        round_keys_neg_prime = self.cipher.expand_key(keys_neg_prime)

        # Encrypt
        ct_neg = self.cipher.encrypt(pt_neg, round_keys_neg, rounds=rounds)
        ct_neg_prime = self.cipher.encrypt(pt_neg_prime, round_keys_neg_prime, rounds=rounds)

        # --- 3. Format Data for Neural Network ---
        # We need to combine CT and CT' into a single feature vector.
        # Structure: [CT_Left, CT_Right, CT_Prime_Left, CT_Prime_Right]
        # Note: self.cipher.encrypt returns [Right, Left] (low, high)

        # For Positive
        # Concatenate: R, L, R', L'
        X_pos = np.concatenate([ct_pos, ct_pos_prime], axis=1)
        Y_pos = np.ones((half_batch, 1), dtype=np.float32)

        # For Negative
        X_neg = np.concatenate([ct_neg, ct_neg_prime], axis=1)
        Y_neg = np.zeros((half_batch, 1), dtype=np.float32)

        # Combine
        X = np.concatenate([X_pos, X_neg], axis=0)
        Y = np.concatenate([Y_pos, Y_neg], axis=0)

        return X, Y

    def convert_to_binary(self, X):
        """
        Converts array of uint16 integers to binary bits for ResNet input.
        Input Shape: (Batch, 4) -> [R, L, R', L']
        Output Shape: (Batch, 4 * 16) -> 64 bits
        """
        # FIX: Force Big-Endian (>u2) so byte 0 is MSB and byte 1 is LSB.
        # This ensures bits are [b15, b14, ... b0] instead of [b7...b0, b15...b8]
        X_big_endian = X.astype('>u2')

        # Unpack bits
        X_bin = np.unpackbits(X_big_endian.view(np.uint8), axis=1)
        return X_bin.astype(np.float32)

class MultiDiffGenerator:
    def __init__(self):
        self.cipher = Simon3264()
        
        # --- Differential A (The one you used) ---
        # Input: (0x0, 0x40), Key: (..., 0x40)
        self.diff_A_in = [0x0040, 0x0000] # [Right, Left]
        self.diff_A_key = [0x0000, 0x0000, 0x0000, 0x0040]

        # --- Differential B (From Table 3 of the paper) ---
        # Input: (0x0, 0x8010), Key: (..., 0x8010)
        self.diff_B_in = [0x8010, 0x0000] # [Right, Left]
        self.diff_B_key = [0x0000, 0x0000, 0x0000, 0x8010]

    def convert_to_binary(self, X):
        # Keep your successful Big-Endian fix
        X_big_endian = X.astype('>u2') 
        return np.unpackbits(X_big_endian.view(np.uint8), axis=1).astype(np.float32)

    def generate_batch(self, batch_size=10000, rounds=8):
        half_batch = batch_size // 2
        quarter_batch = half_batch // 2
        
        # --- 1. Positive Samples (Label 1) ---
        # We split positive samples: 50% use Diff A, 50% use Diff B
        
        # Group A
        keys_A = np.frombuffer(os.urandom(2 * 4 * quarter_batch), dtype=np.uint16).reshape(quarter_batch, 4)
        keys_A_prime = keys_A ^ np.array(self.diff_A_key, dtype=np.uint16)
        pt_A = np.frombuffer(os.urandom(2 * 2 * quarter_batch), dtype=np.uint16).reshape(quarter_batch, 2)
        pt_A_prime = pt_A ^ np.array(self.diff_A_in, dtype=np.uint16)
        
        # Group B
        keys_B = np.frombuffer(os.urandom(2 * 4 * quarter_batch), dtype=np.uint16).reshape(quarter_batch, 4)
        keys_B_prime = keys_B ^ np.array(self.diff_B_key, dtype=np.uint16)
        pt_B = np.frombuffer(os.urandom(2 * 2 * quarter_batch), dtype=np.uint16).reshape(quarter_batch, 2)
        pt_B_prime = pt_B ^ np.array(self.diff_B_in, dtype=np.uint16)
        
        # Encrypt Group A
        rk_A = self.cipher.expand_key(keys_A)
        rk_A_p = self.cipher.expand_key(keys_A_prime)
        ct_A = self.cipher.encrypt(pt_A, rk_A, rounds)
        ct_A_p = self.cipher.encrypt(pt_A_prime, rk_A_p, rounds)
        
        # Encrypt Group B
        rk_B = self.cipher.expand_key(keys_B)
        rk_B_p = self.cipher.expand_key(keys_B_prime)
        ct_B = self.cipher.encrypt(pt_B, rk_B, rounds)
        ct_B_p = self.cipher.encrypt(pt_B_prime, rk_B_p, rounds)
        
        # Combine Positive
        X_pos_A = np.concatenate([ct_A, ct_A_p], axis=1)
        X_pos_B = np.concatenate([ct_B, ct_B_p], axis=1)
        X_pos = np.concatenate([X_pos_A, X_pos_B], axis=0)
        Y_pos = np.ones((half_batch, 1), dtype=np.float32)

        # --- 2. Negative Samples (Label 0) ---
        # Pure Random (Null Hypothesis)
        keys_neg = np.frombuffer(os.urandom(2 * 4 * half_batch), dtype=np.uint16).reshape(half_batch, 4)
        keys_neg_p = np.frombuffer(os.urandom(2 * 4 * half_batch), dtype=np.uint16).reshape(half_batch, 4)
        pt_neg = np.frombuffer(os.urandom(2 * 2 * half_batch), dtype=np.uint16).reshape(half_batch, 2)
        pt_neg_p = np.frombuffer(os.urandom(2 * 2 * half_batch), dtype=np.uint16).reshape(half_batch, 2)
        
        rk_neg = self.cipher.expand_key(keys_neg)
        rk_neg_p = self.cipher.expand_key(keys_neg_p)
        
        ct_neg = self.cipher.encrypt(pt_neg, rk_neg, rounds)
        ct_neg_p = self.cipher.encrypt(pt_neg_p, rk_neg_p, rounds)
        
        X_neg = np.concatenate([ct_neg, ct_neg_p], axis=1)
        Y_neg = np.zeros((half_batch, 1), dtype=np.float32)

        # Shuffle
        X = np.concatenate([X_pos, X_neg], axis=0)
        Y = np.concatenate([Y_pos, Y_neg], axis=0)
        
        perm = np.random.permutation(batch_size)
        return X[perm], Y[perm]

# --- QUICK TEST ---
if __name__ == "__main__":
    gen = SimonDataGenerator()
    X, Y = gen.generate_batch(batch_size=10, rounds=12) # Testing 12 rounds

    print(f"Generated Data Shape: {X.shape}")
    print(f"Sample Positive Entry (Hex):\n {tuple(map(hex, X[0]))}")
    print(f"Label: {Y[0]}")

    X_bin = gen.convert_to_binary(X)
    print(f"Binary Feature Shape: {X_bin.shape}") # Should be (10, 64) since 4 words * 16 bits = 64
