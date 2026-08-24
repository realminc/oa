plugins {
    id("com.android.application") version "8.13.2" apply false
}

val oaRootDirectory = rootDir.resolve("../../..").canonicalFile

layout.buildDirectory.set(
	oaRootDirectory.resolve("build/android/oaMobileLab/Gradle/root"))

subprojects {
	layout.buildDirectory.set(
		oaRootDirectory.resolve("build/android/oaMobileLab/Gradle/$name"))
}
