#include <stdio.h>
#include <string.h>

// Объявления ассемблерных функций
extern int is_palindrome(char* str);    // Проверка на палиндром
extern void rule1_transform(char* str); // Правило 1: обратить правое крыло
extern void rule2_transform(char* str, int len); // Правило 2: длина -> двоичная строка

int main() {
    char input[101];
    printf("Введите текст: ");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';
    printf("Исходный текст: %s\n", input);
    int is_pali = is_palindrome(input);
    char output[101];
    strcpy(output, input); // Копируем исходную строку
    if (is_pali) {
        rule1_transform(output);
        printf("Применено правило 1: обратить правое крыло палиндрома\n");
    } else {
        int len = strlen(input);
        rule2_transform(output, len);
        printf("Применено правило 2: заменить на двоичное представление длины\n");
    }
    printf("Преобразованный текст: %s\n", output);
    return 0;
}