plugins {
    id("com.gradle.enterprise") version "3.19.2"
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.10.0"
}

develocity {
    buildScan {
        termsOfUseUrl = "https://gradle.com/terms-of-service"
        termsOfUseAgree = "yes"
        uploadInBackground = !System.getenv("CI").toBoolean()
//        publishOnFailure()
    }
}

rootProject.name = "field-vmc"
