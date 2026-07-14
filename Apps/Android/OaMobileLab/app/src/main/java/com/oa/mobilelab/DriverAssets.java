package com.oa.mobilelab;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

final class DriverAssets {
    private static final long TURNIP_SIZE = 19_142_520L;

    private DriverAssets() {
    }

    static File prepareTurnip(Context context) throws IOException {
        File directory = new File(context.getFilesDir(), "drivers/turnip-26.1.4");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Could not create " + directory);
        }

        File driver = new File(directory, "libvulkan_freedreno.so");
        if (driver.isFile() && driver.length() == TURNIP_SIZE) {
            return directory;
        }

        File temporary = new File(directory, "libvulkan_freedreno.so.tmp");
        try (InputStream input = context.getAssets().open("turnip/libvulkan_freedreno.so");
             FileOutputStream output = new FileOutputStream(temporary)) {
            input.transferTo(output);
        }
        Files.move(
                temporary.toPath(),
                driver.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE);
        return directory;
    }
}
