#!/usr/bin/env Rscript
# Reference classical calibration via R's `survey` package.
#
# The parity oracle for calibration_paper.classical: given a design matrix and
# target vector, run survey::calibrate with a chosen calibration function and
# emit the g-weights (calibrated / design). The Python test (tests/test_r_parity.py)
# feeds it the SAME problem and asserts the pure-NumPy solvers match R's answer,
# so the paper's comparison is grounded against the field's actual tool rather
# than this reimplementation alone.
#
# I/O contract (kept dead simple so the bridge is a subprocess, not an ABI):
#   argv[1] = calfun: "raking" | "linear" | "logit"
#   argv[2] = path to the matrix CSV  (n_targets rows, n_records columns; no header)
#   argv[3] = path to the target CSV  (one value per line; no header)
#   argv[4] = path to the design-weight CSV (one value per line; no header)
#   argv[5] = path to write the g-weight CSV (one value per line; no header)
#   argv[6] = lower g-weight bound (only for calfun="logit"), optional
#   argv[7] = upper g-weight bound (only for calfun="logit"), optional
#
# The design matrix rows are the calibration variables; survey::calibrate wants
# them as columns of a model matrix, so we transpose. survey solves
# sum_k w_k x_k = population exactly (to its epsilon), the same calibration
# equations the Python solvers solve.

suppressMessages(library(survey))

args <- commandArgs(trailingOnly = TRUE)
calfun <- args[[1]]
matrix_path <- args[[2]]
target_path <- args[[3]]
weight_path <- args[[4]]
out_path <- args[[5]]

# A: n_targets x n_records. mm: n_records x n_targets (survey's model matrix).
A <- as.matrix(read.csv(matrix_path, header = FALSE))
population <- scan(target_path, quiet = TRUE)
w0 <- scan(weight_path, quiet = TRUE)
mm <- t(A)
n_records <- nrow(mm)
n_targets <- ncol(mm)

# Build a design whose data frame carries the calibration variables v1..vT, and
# a calibration formula with NO intercept (every target is an explicit column,
# so the constant "total" target is just one of the vN columns when the caller
# put a row of ones in A). This keeps the model matrix equal to `mm` exactly.
colnames(mm) <- paste0("v", seq_len(n_targets))
dat <- as.data.frame(mm)
dat$.id <- seq_len(n_records)
dat$.w0 <- w0
design <- svydesign(ids = ~.id, weights = ~.w0, data = dat)
form <- as.formula(paste("~ 0 +", paste(colnames(mm), collapse = " + ")))
names(population) <- colnames(mm)

if (calfun == "logit") {
  bounds <- c(as.numeric(args[[6]]), as.numeric(args[[7]]))
  calibrated <- calibrate(design, form, population = population,
                          calfun = "logit", bounds = bounds)
} else {
  calibrated <- calibrate(design, form, population = population, calfun = calfun)
}

g <- weights(calibrated) / w0
write.table(g, out_path, row.names = FALSE, col.names = FALSE)
