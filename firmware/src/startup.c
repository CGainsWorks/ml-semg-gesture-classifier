/*
 * Minimal Cortex-M4F start-up: vector table, .data and .bss initialisation,
 * FPU enable, C++ constructors, then main().
 *
 * No vendor HAL. The firmware needs a UART and a cycle counter and both are
 * a handful of register writes (board.c).
 */

#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern void __libc_init_array(void);
extern int main(void);

void Reset_Handler(void);
void Default_Handler(void);

void Default_Handler(void) {
  for (;;) {
  }
}

void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

/* Core exceptions only: the firmware enables no peripheral interrupts. */
__attribute__((section(".isr_vector"), used)) void (*const g_vectors[16])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0,
    0,
    0,
    0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};

#define SCB_CPACR (*(volatile uint32_t *)0xE000ED88u)

void Reset_Handler(void) {
  uint32_t *src = &_sidata;
  for (uint32_t *dst = &_sdata; dst < &_edata;) {
    *dst++ = *src++;
  }
  for (uint32_t *dst = &_sbss; dst < &_ebss;) {
    *dst++ = 0;
  }

  /* CP10 and CP11 full access: the code is built for the hard-float ABI. */
  SCB_CPACR |= (0xFu << 20);
  __asm volatile("dsb\n\tisb" ::: "memory");

  __libc_init_array();
  main();
  for (;;) {
  }
}
