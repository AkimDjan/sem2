# Правило 1: обратить правое крыло палиндрома
# Пример: "123321" -> "123123"

.globl rule1_transform
rule1_transform:
    movq %rdi, %rsi       # rsi = начало строки
    call strlen           # Длина строки в rax

    movq %rax, %rcx       # rcx = длина
    shrq $1, %rcx         # Делим длину на 2 (середина)
    leaq (%rsi, %rcx), %rdi # rdi = начало правого крыла

    # Обращение правой части
reverse_loop:
    movq %rdi, %rdx       # Копия правого крыла
    movq %rsi, %rax
    addq %rcx, %rax       # Конец строки
reverse:
    cmpq %rdx, %rax
    jle  end_reverse
    movb (%rdx), %bl
    movb (%rax), %cl
    movb %cl, (%rdx)
    movb %bl, (%rax)
    incq %rdx
    decq %rax
    jmp reverse

end_reverse:
    ret

strlen:
    xorq %rax, %rax
strlen_loop:
    cmpb $0, (%rsi, %rax)
    je   strlen_end
    incq %rax
    jmp strlen_loop
strlen_end:
    ret