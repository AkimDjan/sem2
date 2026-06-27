	.file	"get_flags.c"
	.text
	.globl	get_flags
	.type	get_flags, @function
get_flags:
	endbr64
	pushfq            # Сохраняем регистр флагов в стек
    movq (%rsp), %rax # Загружаем значение из стека в rax
    pop %rcx          # Очищаем стек
	ret
	.size	get_flags, .-get_flags
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
