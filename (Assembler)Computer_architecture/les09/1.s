	.file	"1.c"
	.text
	.globl	mult
	.def	mult;	.scl	2;	.type	32;	.endef
mult:
	pushq	%rbp
	movq	%rsp, %rbp
	andq	$-16, %rsp
	imulq	%r8, %rdx
	movq	%rdx, (%rcx)
	movq	%rdx, %rax
	sarq	$63, %rax
	movq	%rax, 8(%rcx)
	leave
	ret
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
