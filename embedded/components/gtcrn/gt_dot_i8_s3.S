/*
 * ESP32-S3 int8 dot product - optimized fused-load version
 *
 * int32_t gt_dot_i8_s3(const int8_t *a,
 *                      const int8_t *b,
 *                      int n);
 *
 * Computes:
 *
 *     sum(a[i] * b[i])
 *
 * Vector path:
 *     16 signed int8 multiplies per VMULAS
 *
 * Optimization:
 *     EE.VMULAS.S8.ACCX.LD.IP
 *     performs the MAC while loading the next b block.
 *
 * The final block is handled separately so we NEVER read past
 * the end of the arrays.
 */

.text
.align 4

.global gt_dot_i8_s3
.type gt_dot_i8_s3, @function

gt_dot_i8_s3:

    /*
     * Xtensa windowed ABI:
     *
     * a2 = a
     * a3 = b
     * a4 = n
     */

    entry a1, 16

    /*
     * Clear 40-bit ACCX accumulator.
     */
    ee.zero.accx

    /*
     * blocks = n / 16
     */
    srli a5, a4, 4

    /*
     * No complete SIMD block?
     */
    beqz a5, .L_remainder

    /*
     * Load first 16-byte block from each vector.
     */
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16

    /*
     * We already have block 0 loaded.
     *
     * Process blocks 0 ... blocks-2 using the fused
     * multiply/load instruction.
     *
     * After each iteration:
     *
     *     q0 = next block of a
     *     q1 = next block of b
     *
     * The final block is handled separately.
     */

    addi a5, a5, -1

    /*
     * Only one SIMD block total.
     */
    beqz a5, .L_last_vector

.L_vector_loop:

    /*
     * ACCX += q0 * q1
     *
     * At the same time:
     *
     *     q1 <- next 16 bytes of b
     */
    ee.vmulas.s8.accx.ld.ip q1, a3, 16, q0, q1

    /*
     * Load the matching next 16 bytes of a.
     */
    ee.vld.128.ip q0, a2, 16

    addi a5, a5, -1
    bnez a5, .L_vector_loop

.L_last_vector:

    /*
     * Accumulate the final preloaded block.
     */
    ee.vmulas.s8.accx q0, q1

.L_remainder:

    /*
     * Extract low 32 bits of the 40-bit accumulator.
     */
    rur.accx_0 a5

    /*
     * remainder = n % 16
     */
    movi a6, 15
    and a4, a4, a6

    beqz a4, .L_done

.L_scalar_loop:

    /*
     * Load signed bytes.
     */
    l8ui a6, a2, 0
    l8ui a7, a3, 0

    /*
     * Convert uint8 representation to signed int8.
     */
    sext a6, a6, 7
    sext a7, a7, 7

    /*
     * Signed 32-bit multiply.
     */
    mull a6, a6, a7

    /*
     * Add remainder product.
     */
    add a5, a5, a6

    /*
     * Advance pointers.
     */
    addi a2, a2, 1
    addi a3, a3, 1

    /*
     * Next remainder element.
     */
    addi a4, a4, -1
    bnez a4, .L_scalar_loop

.L_done:

    /*
     * Return int32 result in a2.
     */
    mov a2, a5

    retw.n

.size gt_dot_i8_s3, .-gt_dot_i8_s3
