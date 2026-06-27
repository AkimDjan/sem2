	.file	"exc2.c"
	.text
	.globl	absdiff
	.def	absdiff;	.scl	2;	.type	32;	.endef
absdiff:
	cmpl	%edx, %ecx
	jge	.L2
	movl	%edx, %eax
	subl	%ecx, %eax
	ret
.L2:
	movl	%ecx, %eax
	subl	%edx, %eax
	ret
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
