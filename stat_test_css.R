strong <- read.csv(file = "data/css/data_strong.csv", header = TRUE, row.names = 1)

mean_strong_zero_kn <- mean(strong$zero-strong$zero_kn) # > 0 means we're doing better.
mean_strong_zero_mcs <- mean(strong$zero-strong$zero_mcs)
mean_strong_max_kn <- mean(strong$max-strong$max_kn)
mean_strong_max_mcs <- mean(strong$max-strong$max_mcs)

shapiro.test(strong$zero_kn - strong$zero)
shapiro.test(strong$zero_mcs - strong$zero)
shapiro.test(strong$max_kn - strong$max)
shapiro.test(strong$max_mcs - strong$max)

wilcox.test(strong$zero, strong$zero_kn, paired = TRUE, alternative = "greater")
wilcox.test(strong$zero, strong$zero_mcs, paired = TRUE, alternative = "less")
wilcox.test(strong$max, strong$max_kn, paired = TRUE, alternative = "greater")
wilcox.test(strong$max, strong$max_mcs, paired = TRUE, alternative = "greater")

###

weak <- read.csv(file = "data/css/data_weak.csv", header = TRUE, row.names = 1)

mean_weak_zero_kn <- mean(weak$zero - weak$zero_kn)
mean_weak_zero_mcs <- mean(weak$zero - weak$zero_mcs)
mean_weak_max_kn <- mean(weak$max - weak$max_kn)
mean_weak_max_mcs <- mean(weak$max - weak$max_mcs)

shapiro.test(weak$zero_kn - weak$zero)
shapiro.test(weak$zero_mcs - weak$zero)
shapiro.test(weak$max_kn - weak$max)
shapiro.test(weak$max_mcs - weak$max)

wilcox.test(weak$zero, weak$zero_kn, paired = TRUE, alternative = "less")
wilcox.test(weak$zero, weak$zero_mcs, paired = TRUE, alternative = "less")
wilcox.test(weak$max, weak$max_kn, paired = TRUE, alternative = "less")
wilcox.test(weak$max, weak$max_mcs, paired = TRUE, alternative = "greater")

###

uniform <- read.csv(file = "data/css/data_uniform.csv", header = TRUE, row.names = 1)

mean_uniform_zero_kn <- mean(uniform$zero - uniform$zero_kn)
mean_uniform_zero_mcs <- mean(uniform$zero - uniform$zero_mcs)
mean_uniform_max_kn <- mean(uniform$max - uniform$max_kn)
mean_uniform_max_mcs <- mean(uniform$max - uniform$max_mcs)

shapiro.test(uniform$zero_kn - uniform$zero)
shapiro.test(uniform$zero_mcs - uniform$zero)
shapiro.test(uniform$max_kn - uniform$max)
shapiro.test(uniform$max_mcs - uniform$max)

wilcox.test(uniform$zero, uniform$zero_kn, paired = TRUE, alternative = "less")
wilcox.test(uniform$zero, uniform$zero_mcs, paired = TRUE, alternative = "less")
wilcox.test(uniform$max, uniform$max_kn, paired = TRUE, alternative = "less")
wilcox.test(uniform$max, uniform$max_mcs, paired = TRUE, alternative = "greater")

###

intermediate <- read.csv(file = "data/css/data_intermediate.csv", header = TRUE, row.names = 1)

mean_intermediate_zero_kn <- mean(intermediate$zero - intermediate$zero_kn)
mean_intermediate_zero_mcs <- mean(intermediate$zero - intermediate$zero_mcs)
mean_intermediate_max_kn <- mean(intermediate$max - intermediate$max_kn)
mean_intermediate_max_mcs <- mean(intermediate$max - intermediate$max_mcs)

shapiro.test(intermediate$zero_kn - intermediate$zero)
shapiro.test(intermediate$zero_mcs - intermediate$zero)
shapiro.test(intermediate$max_kn - intermediate$max)
shapiro.test(intermediate$max_mcs - intermediate$max)

wilcox.test(intermediate$zero, intermediate$zero_kn, paired = TRUE, alternative = "less")
wilcox.test(intermediate$zero, intermediate$zero_mcs, paired = TRUE, alternative = "less")
wilcox.test(intermediate$max, intermediate$max_kn, paired = TRUE, alternative = "less")
wilcox.test(intermediate$max, intermediate$max_mcs, paired = TRUE, alternative = "greater")

###

mean_strong_zero <- mean(strong$classic - strong$zero)
mean_strong_max <- mean(strong$classic - strong$max)
wilcox.test(strong$classic, strong$zero, paired = TRUE, alternative = "greater")
wilcox.test(strong$classic, strong$max, paired = TRUE, alternative = "less")

mean_weak_zero <- mean(weak$classic - weak$zero)
mean_weak_max <- mean(weak$classic - weak$max)
wilcox.test(weak$classic, weak$zero, paired = TRUE, alternative = "greater")
wilcox.test(weak$classic, weak$max, paired = TRUE, alternative = "less")

mean_uniform_zero <- mean(uniform$classic - uniform$zero)
mean_uniform_max <- mean(uniform$classic - uniform$max)
wilcox.test(uniform$classic, uniform$zero, paired = TRUE, alternative = "greater")
wilcox.test(uniform$classic, uniform$max, paired = TRUE, alternative = "less")

mean_intermediate_zero <- mean(intermediate$classic - intermediate$zero)
mean_intermediate_max <- mean(intermediate$classic - intermediate$max)
wilcox.test(intermediate$classic, intermediate$zero, paired = TRUE, alternative = "greater")
wilcox.test(intermediate$classic, intermediate$max, paired = TRUE, alternative = "less")

