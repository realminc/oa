plugins {
	id("com.android.application")
}

val oaRootDirectory = rootProject.file("../../..").canonicalFile
val turnipAssetsDirectory =
	oaRootDirectory.resolve("Build/Android/OaMobileLab/Assets")

val fetchTurnip by tasks.registering(Exec::class) {
	group = "oa android"
	description = "Fetches and verifies the pinned Mesa Turnip driver"
	commandLine("bash", rootProject.file("tools/fetch-turnip.sh").absolutePath)
	inputs.property("turnipVersion", "26.1.4")
	outputs.file(turnipAssetsDirectory.resolve("turnip/libvulkan_freedreno.so"))
}

android {
	namespace = "com.oa.mobilelab"
	compileSdk = 36
	ndkVersion = "29.0.14206865"

	sourceSets {
		getByName("main").assets.srcDir(turnipAssetsDirectory)
	}

	defaultConfig {
		applicationId = "com.oa.mobilelab"
		minSdk = 33
		targetSdk = 36
		versionCode = 1
		versionName = "0.3.0-lab"

		ndk {
			abiFilters += "arm64-v8a"
		}

		externalNativeBuild {
			cmake {
				cppFlags += listOf("-std=c++20", "-Wall", "-Wextra", "-Wpedantic")
				arguments += listOf("-DANDROID_STL=c++_shared")
			}
		}
	}

	packaging {
		jniLibs.useLegacyPackaging = true
	}

	buildTypes {
		debug {
			isDebuggable = true
			applicationIdSuffix = ".debug"
			versionNameSuffix = "-debug"
		}
		release {
			isMinifyEnabled = false
		}
	}

	compileOptions {
		sourceCompatibility = JavaVersion.VERSION_17
		targetCompatibility = JavaVersion.VERSION_17
	}

	externalNativeBuild {
		cmake {
			path = file("src/main/cpp/CMakeLists.txt")
			version = "4.1.2"
			buildStagingDirectory = oaRootDirectory.resolve(
				"Build/Android/OaMobileLab/CMake")
		}
	}
}

tasks.named("preBuild").configure {
	dependsOn(fetchTurnip)
}

listOf("debug", "release").forEach { buildType ->
	val buildTypeName = buildType.replaceFirstChar(Char::uppercaseChar)
	val stageApk = tasks.register<Copy>("stage${buildTypeName}Apk") {
		from(layout.buildDirectory.dir("outputs/apk/$buildType"))
		include("*.apk")
		into(oaRootDirectory.resolve("Bin/Android/OaMobileLab"))
		rename { "OaMobileLab-$buildType.apk" }
	}
	tasks.matching { it.name == "assemble$buildTypeName" }.configureEach {
		finalizedBy(stageApk)
	}
}
