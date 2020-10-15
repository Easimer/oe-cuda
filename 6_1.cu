#include <stdio.h>

static void encrypt(unsigned char *cipher, unsigned char const *clear, int clear_len, unsigned char const *key, int key_len) {
    int key_cur = 0;
    for(int i = 0; i < clear_len; i++) {
        auto key_byte = key[key_cur];

        cipher[i] = clear[i] ^ key_byte;

        key_cur++;
        if(key_cur == key_len) {
            key_cur = 0;
        }
    }
}

static void decrypt(unsigned char *clear, unsigned char const *cipher, int clear_len, unsigned char const *key, int key_len) {
    int key_cur = 0;
    for(int i = 0; i < clear_len; i++) {
        auto key_byte = key[key_cur];

        clear[i] = cipher[i] ^ key_byte;

        key_cur++;
        if(key_cur == key_len) {
            key_cur = 0;
        }
    }
}

static void generate_key(unsigned char* key, int key_len) {
    for(int i = 0; i < key_len; i++) {
        key[i] = rand() & 0xFF;
    }
}

static void crack_key(unsigned char *key, unsigned char const *ciphertext, unsigned char const *cleartext, int msg_len) {
    // TODO:
}

constexpr int KEY_LEN = 128;
constexpr int MSG_LEN = 256;

static void generate_message(char *msg, int msg_len, char const *rep, int rep_len) {
    int rep_cur = 0;

    for(int i = 0; i < msg_len; i++) {
        msg[i] = rep[rep_cur];

        rep_cur++;
        if(rep_cur == rep_len) {
            rep_cur = 0;
        }
    }
}

int main(int argc, char **argv) {
    srand(0);

    auto key = new unsigned char[KEY_LEN];
    generate_key(key, KEY_LEN);

    auto msg = new char[MSG_LEN];
    auto cipher = new char[MSG_LEN];
    auto src = "hajnalban tamadunk ";
    generate_message(msg, MSG_LEN, src, strlen(src)); 

    printf("Message: %.*s\nKey: '%.*s'\n", MSG_LEN, msg, KEY_LEN, key);

    encrypt((unsigned char*)cipher, (unsigned char*)msg, MSG_LEN, (unsigned char*)key, KEY_LEN);

    printf("Ciphertext:\n'%.*s'\n", MSG_LEN, cipher);
    return 0;
}
