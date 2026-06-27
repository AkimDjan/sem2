	.file	"main.cpp"
	.text
	.section .rdata,"dr"
.LC0:
	.ascii "2*3=\0"
.LC1:
	.ascii "\12\0"
	.text
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
main:
	subq	$56, %rsp
	call	__main
	movq	$0, 40(%rsp)
	leaq	40(%rsp), %r8
	movl	$3, %edx
	movl	$2, %ecx
	call	multstore
	leaq	.LC0(%rip), %rdx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rcx
	movq	40(%rsp), %rdx
	call	_ZNSo9_M_insertIxEERSoT_
	movq	%rax, %rcx
	leaq	.LC1(%rip), %rdx
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$0, %eax
	addq	$56, %rsp
	ret
	.globl	mult
	.def	mult;	.scl	2;	.type	32;	.endef
mult:
	movq	%rcx, %rax
	imulq	%rdx, %rax
	ret
	.def	__main;	.scl	2;	.type	32;	.endef
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-Builds project) 14.2.0"
	.def	multstore;	.scl	2;	.type	32;	.endef
	.def	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc;	.scl	2;	.type	32;	.endef
	.def	_ZNSo9_M_insertIxEERSoT_;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
