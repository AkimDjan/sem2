# Проверка, является ли строка палиндромом
# Возвращает 1 (да) или 0 (нет)

.globl is_palindrome
is_palindrome:
    movq %rdi, %rsi       # rsi = начало строки
    movq %rdi, %rdx       # rdx = начало для поиска конца

find_end:
    cmpb $0, (%rdx)       # Ищем конец строки (нулевой символ)
    je   compare
    incq %rdx
    jmp find_end

compare:
    decq %rdx             # Указатель на последний символ (не '\0')
loop:
    cmpq %rdi, %rdx       # Сравниваем адреса (начало и конец)
    jle  is_pali          # Если указатели встретились, это палиндром
    movb (%rdi), %al
    cmpb %al, (%rdx)
    jne  not_pali         # Символы не совпали
    incq %rdi
    decq %rdx
    jmp loop

is_pali:
    movl $1, %eax         # Возвращаем 1
    ret

not_pali:
    movl $0, %eax         # Возвращаем 0
    ret