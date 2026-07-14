package com.oa.mobilelab;

import android.os.Bundle;
import android.os.Handler;
import android.os.ResultReceiver;

final class ProbeResultReceiver extends ResultReceiver {
    interface Listener {
        void onProbeResult(int resultCode, Bundle resultData);
    }

    private final Listener listener;

    ProbeResultReceiver(Handler handler, Listener listener) {
        super(handler);
        this.listener = listener;
    }

    @Override
    protected void onReceiveResult(int resultCode, Bundle resultData) {
        listener.onProbeResult(resultCode, resultData);
    }
}
