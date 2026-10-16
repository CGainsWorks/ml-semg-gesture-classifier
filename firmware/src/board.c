/*
 * USART2 and the DWT cycle counter, by register.
 *
 * Register addresses are from the STM32F411 reference manual (RM0383) and the
 * Armv7-M architecture reference manual. Clock and pin set-up is omitted: the
 * emulator does not model the RCC gating or the GPIO alternate functions and
 * a board port would add both (see benchmarks/README.md).
 */

#include "board.h"

#define USART2_BASE 0x40004400u
#define USART_SR (*(volatile uint32_t *)(USART2_BASE + 0x00u))
#define USART_DR (*(volatile uint32_t *)(USART2_BASE + 0x04u))
#define USART_BRR (*(volatile uint32_t *)(USART2_BASE + 0x08u))
#define USART_CR1 (*(volatile uint32_t *)(USART2_BASE + 0x0Cu))
#define USART_SR_TXE (1u << 7)
#define USART_CR1_UE (1u << 13)
#define USART_CR1_TE (1u << 3)

#define DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define DEMCR_TRCENA (1u << 24)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DWT_CTRL_CYCCNTENA (1u << 0)

void board_uart_init(void) {
  /* 115200 baud from a 16 MHz APB1 clock (HSI at reset). */
  USART_BRR = 0x8Bu;
  USART_CR1 = USART_CR1_UE | USART_CR1_TE;
}

void board_uart_write(const char *text) {
  for (; *text != '\0'; ++text) {
    while ((USART_SR & USART_SR_TXE) == 0u) {
    }
    USART_DR = (uint32_t)(unsigned char)*text;
  }
}

void board_cycle_counter_init(void) {
  DEMCR |= DEMCR_TRCENA;
  DWT_CYCCNT = 0u;
  DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

uint32_t board_cycles(void) { return DWT_CYCCNT; }
