	.file	"exc3.c"
	.text
	.globl	loop_do_while
	.def	loop_do_while;	.scl	2;	.type	32;	.endef

	
loop_do_while:
	movl	%ecx, %eax
	imull	%ecx, %ecx
	leal	0(,%rax,4), %edx
	.p2align 4
.L2:
	leal	5(%rcx,%rax), %eax
	subl	$2, %edx
	testl	%edx, %edx
	jg	.L2
	ret


	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
