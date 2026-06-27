#include <stdio.h>
#include <stdint.h>

int64_t test();
int64_t get_flags();

int64_t print_flags(int64_t flags) 
{
	// Извлекаем значения отдельных флагов
	int cf = flags >> 0 & 1;	// Carry Flag (0-й бит)
	int zf = flags >> 6 & 1;	// Zero Flag (6-й бит)
	int sf = flags >> 7 & 1;	// Sign Flag (7-й бит)
	int of = flags >> 11 & 1;	// Overflow Flag (11-й бит)

	// Выводим результат
	printf("Состояние регистров флагов:\n");
	printf("CF (Carry Flag): %d\n", cf);
	printf("ZF (Zero Flag): %d\n", zf);
	printf("SF (Sign Flag): %d\n", sf);
	printf("OF (Overflow Flag): %d\n", of);
}

int main() 
{
	int64_t i = test();
	int64_t flags = get_flags();

	print_flags(flags);

	printf("dec: %ld\n", i);
	printf("hex: 0x%016lX\n", i);
	return 0;
}