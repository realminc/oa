plugins {
    id("com.android.application") version "8.13.2" apply false
}

val oaRootDirectory = rootDir.resolve("../../..").canonicalFile

layout.buildDirectory.set(
	oaRootDirectory.resolve("Build/Android/OaMobileLab/Gradle/root"))

subprojects {
	layout.buildDirectory.set(
		oaRootDirectory.resolve("Build/Android/OaMobileLab/Gradle/$name"))
}
