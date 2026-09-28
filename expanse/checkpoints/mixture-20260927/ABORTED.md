This preparation was stopped before the first optimizer update. Review found
that scorer v1 could mistake multiplication `x` for an answer marker and accept
merged tokens such as `answer42`. No model generations were evaluated and no
candidate checkpoint was completed. The original manifest, logs and cache are
preserved. The corrected scorer and replacement study are versioned separately.
