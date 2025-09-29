FROM eclipse-temurin:25
RUN mkdir -p /experiment
WORKDIR /experiment
ENV GRADLE_TASK=runClassicVMCBatch
ENV MAX_SEED=500
COPY effects effects
COPY gradle gradle
COPY src src
COPY docker docker
COPY *.kts ./
COPY *.properties ./
COPY gradlew* ./
RUN sed -i '/alias(libs.plugins.gitSemVer)/d' build.gradle.kts
RUN sed -i '/alias(libs.plugins.multiJvmTesting)/d' build.gradle.kts
RUN sed -i '/multiJvm {/,/}/d' build.gradle.kts
CMD MAX_SEED=$MAX_SEED ./gradlew $GRADLE_TASK

