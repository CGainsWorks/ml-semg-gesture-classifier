/* The two things the harness needs from the chip: a serial port and a clock. */

#ifndef SEMG_FIRMWARE_BOARD_H_
#define SEMG_FIRMWARE_BOARD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* USART2, the port an STM32F411 Nucleo routes to its debug probe. */
void board_uart_init(void);
void board_uart_write(const char *text);

/* DWT cycle counter. Under Renode it advances with executed instructions. */
void board_cycle_counter_init(void);
uint32_t board_cycles(void);

#ifdef __cplusplus
}
#endif

#endif /* SEMG_FIRMWARE_BOARD_H_ */
