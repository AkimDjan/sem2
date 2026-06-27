	.file	"excersize8.c"
	.text
	.globl	IsZero
	.def	IsZero;	.scl	2;	.type	32;	.endef
IsZero:
	testl	%edx, %edx
	js	.L4
	movl	$0, (%rcx)
	ret
.L4:
	movl	$1, (%rcx)
	ret
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
