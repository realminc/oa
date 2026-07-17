plugins {
	id("com.android.application")
}

val oaRootDirectory = rootProject.file("../../..").canonicalFile
val turnipAssetsDirectory =
	oaRootDirectory.resolve("Build/Android/OaMobileLab/Assets")

val releaseSigningEnvironment = mapOf(
	"storeFile" to providers.environmentVariable("OA_ANDROID_KEYSTORE").orNull,
	"storePassword" to providers.environmentVariable("OA_ANDROID_KEYSTORE_PASSWORD").orNull,
	"keyAlias" to providers.environmentVariable("OA_ANDROID_KEY_ALIAS").orNull,
	"keyPassword" to providers.environmentVariable("OA_ANDROID_KEY_PASSWORD").orNull,
)
val configuredReleaseSigningValues = releaseSigningEnvironment.values.count { !it.isNullOrBlank() }
if (configuredReleaseSigningValues != 0 && configuredReleaseSigningValues != releaseSigningEnvironment.size) {
	throw GradleException(
		"Release signing requires OA_ANDROID_KEYSTORE, OA_ANDROID_KEYSTORE_PASSWORD, " +
			"OA_ANDROID_KEY_ALIAS, and OA_ANDROID_KEY_PASSWORD",
	)
}

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
		versionCode = 2
		versionName = "0.4.0-build-week"

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

	signingConfigs {
		if (configuredReleaseSigningValues == releaseSigningEnvironment.size) {
			create("release") {
				storeFile = file(releaseSigningEnvironment.getValue("storeFile")!!)
				storePassword = releaseSigningEnvironment.getValue("storePassword")
				keyAlias = releaseSigningEnvironment.getValue("keyAlias")
				keyPassword = releaseSigningEnvironment.getValue("keyPassword")
			}
		}
	}

	buildTypes {
		debug {
			isDebuggable = true
			applicationIdSuffix = ".debug"
			versionNameSuffix = "-debug"
		}
		release {
			isMinifyEnabled = false
			signingConfig = signingConfigs.findByName("release")
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
