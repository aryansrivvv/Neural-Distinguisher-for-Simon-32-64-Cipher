class Simon3264:
    def __init__(self):
        # Simon 32/64 Constants
        self.n = 16  # Word size
        self.m = 4   # Number of key words
        self.rounds = 32

        # Z0 sequence for Simon 32/64 key schedule
        self.z0 = 0b11111010001001010110000111001101111101000100101011000011100110

    def _rol(self, x, k):
        """Circular rotate left x by k bits."""
        return ((x << k) & 0xFFFF) | ((x >> (16 - k)) & 0xFFFF)

    def _ror(self, x, k):
        """Circular rotate right x by k bits."""
        return ((x >> k) & 0xFFFF) | ((x << (16 - k)) & 0xFFFF)

    def expand_key(self, master_key):
        """
        Expands a 64-bit master key into 32 round keys.
        Input: master_key (int) or numpy array of shape (batch_size, 4)
               Each element must be a 16-bit word.
        """
        # Handle single int key input (split into 4 words)
        if isinstance(master_key, int):
            k = [(master_key >> (16 * i)) & 0xFFFF for i in range(4)]
        else:
            # Assume numpy input for batch processing
            k = [master_key[:, i] for i in range(4)]

        key_schedule = []

        # The first m round keys are the master key words directly
        for i in range(4):
            key_schedule.append(k[i])

        # Generate remaining round keys
        for i in range(self.m, self.rounds):
            tmp = self._ror(k[i-1], 3)
            if self.m == 4:
                tmp = tmp ^ k[i-3]
            tmp = tmp ^ self._ror(tmp, 1)

            # Get the constant z_j bit
            z_bit = (self.z0 >> ((61 - (i - self.m)) % 62)) & 1

            k_new = (~k[i-self.m]) & 0xFFFF # Bitwise NOT
            k_new = k_new ^ tmp ^ z_bit ^ 3

            k.append(k_new)
            key_schedule.append(k_new)

        return key_schedule

    def encrypt(self, plaintext, keys, rounds=32):
        """
        Encrypts plaintext using the generated round keys.
        Input:
            plaintext: numpy array of shape (batch_size, 2) -> (Left, Right)
            keys: list of 32 round keys (from expand_key)
            rounds: number of rounds to encrypt (useful for partial encryption)
        """
        # Unpack Left and Right words (16-bit each)
        # Plaintext is typically [Left_Word, Right_Word]
        l = plaintext[:, 1] # High word
        r = plaintext[:, 0] # Low word

        for i in range(rounds):
            # Round function:
            # L_new = R ^ ((L << 1) & (L << 8)) ^ (L << 2) ^ k
            # R_new = L

            f_l = (self._rol(l, 1) & self._rol(l, 8)) ^ self._rol(l, 2)
            new_l = r ^ f_l ^ keys[i]

            r = l
            l = new_l

        # Return combined ciphertext pairs (Right, Left) to match standard test vectors
        return np.stack([r, l], axis=1)

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Test Vector for Simon 32/64 from NSA paper or standard sources
    # Key: 0x1918 0x1110 0x0908 0x0100
    # Plaintext: 0x6565 0x6877
    # Expected Ciphertext: 0xc69b 0xe9bb

    simon = Simon3264()

    # 1. Setup Key (Batch size 1 for testing)
    key_batch = np.array([[0x0100, 0x0908, 0x1110, 0x1918]], dtype=np.uint16)
    round_keys = simon.expand_key(key_batch)

    # 2. Setup Plaintext
    # 0x6565 (25957) and 0x6877 (26743)
    pt_batch = np.array([[0x6877, 0x6565]], dtype=np.uint16)

    # 3. Encrypt
    ct = simon.encrypt(pt_batch, round_keys)

    print(f"Computed Ciphertext: {hex(ct[0][1])} {hex(ct[0][0])}")
    print("Expected Ciphertext: 0xc69b 0xe9bb")

    if ct[0][1] == 0xc69b and ct[0][0] == 0xe9bb:
        print("SUCCESS: Cipher implementation works correctly.")
    else:
        print("FAILURE: Check bitwise operations.")
