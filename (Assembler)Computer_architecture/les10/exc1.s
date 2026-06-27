	.file	"exc1.c"
	.text
	.globl	cond
	.def	cond;	.scl	2;	.type	32;	.endef
cond:
	testq	%rdx, %rdx
	je	.L1
	cmpl	%ecx, (%rdx)
	jge	.L1
	movl	%ecx, (%rdx)
.L1:
	ret
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
