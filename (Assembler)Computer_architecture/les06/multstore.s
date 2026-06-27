	.file	"multstore.c"
	.text
	.globl	mult
	.def	mult;	.scl	2;	.type	32;	.endef
mult:
	movq	%rcx, %rax
	imulq	%rdx, %rax
	ret
	.globl	multstore
	.def	multstore;	.scl	2;	.type	32;	.endef
multstore:
	pushq	%rbx
	subq	$32, %rsp
	movq	%r8, %rbx
	call	mult
	cltq
	movq	%rax, (%rbx)
	addq	$32, %rsp
	popq	%rbx
	ret
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
