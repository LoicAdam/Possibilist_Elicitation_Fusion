strong <- read.csv(file = "data/random/data_strong.csv", header = TRUE, row.names = 1)

mean_strong_zero_kn <- mean(strong$zero-strong$zero_kn) # > 0 means we're doing better.
mean_strong_zero_mcs <- mean(strong$zero-strong$zero_mcs)
mean_strong_max_kn <- mean(strong$max-strong$max_kn)
mean_strong_max_mcs <- mean(strong$max-strong$max_mcs)

shapiro.test(strong$zero_kn - strong$zero)
shapiro.test(strong$zero_mcs - strong$zero)
shapiro.test(strong$max_kn - strong$max)
shapiro.test(strong$max_mcs - strong$max)

wilcox.test(strong$zero, strong$zero_kn, paired = TRUE, alternative = "greater")
wilcox.test(strong$zero, strong$zero_mcs, paired = TRUE, alternative = "greater")
wilcox.test(strong$max, strong$max_kn, paired = TRUE, alternative = "greater")
wilcox.test(strong$max, strong$max_mcs, paired = TRUE, alternative = "greater")

###

weak <- read.csv(file = "data/random/data_weak.csv", header = TRUE, row.names = 1)

mean_weak_zero_kn <- mean(weak$zero - weak$zero_kn)
mean_weak_zero_mcs <- mean(weak$zero - weak$zero_mcs)
mean_weak_max_kn <- mean(weak$max - weak$max_kn)
mean_weak_max_mcs <- mean(weak$max - weak$max_mcs)

shapiro.test(weak$zero_kn - weak$zero)
shapiro.test(weak$zero_mcs - weak$zero)
shapiro.test(weak$max_kn - weak$max)
shapiro.test(weak$max_mcs - weak$max)

wilcox.test(weak$zero, weak$zero_kn, paired = TRUE, alternative = "less")
wilcox.test(weak$zero, weak$zero_mcs, paired = TRUE, alternative = "greater")
wilcox.test(weak$max, weak$max_kn, paired = TRUE, alternative = "less")
wilcox.test(weak$max, weak$max_mcs, paired = TRUE, alternative = "greater")

###

uniform <- read.csv(file = "data/random/data_uniform.csv", header = TRUE, row.names = 1)

mean_uniform_zero_kn <- mean(uniform$zero - uniform$zero_kn)
mean_uniform_zero_mcs <- mean(uniform$zero - uniform$zero_mcs)
mean_uniform_max_kn <- mean(uniform$max - uniform$max_kn)
mean_uniform_max_mcs <- mean(uniform$max - uniform$max_mcs)

shapiro.test(uniform$zero_kn - uniform$zero)
shapiro.test(uniform$zero_mcs - uniform$zero)
shapiro.test(uniform$max_kn - uniform$max)
shapiro.test(uniform$max_mcs - uniform$max)

wilcox.test(uniform$zero, uniform$zero_kn, paired = TRUE, alternative = "greater")
wilcox.test(uniform$zero, uniform$zero_mcs, paired = TRUE, alternative = "greater")
wilcox.test(uniform$max, uniform$max_kn, paired = TRUE, alternative = "greater")
wilcox.test(uniform$max, uniform$max_mcs, paired = TRUE, alternative = "greater")

