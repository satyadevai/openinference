package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.TraceConfig;
import java.net.URI;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.content.Media;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;

class MediaContentSupportTest {

    private static Media byteImage() {
        return new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(new byte[] {1, 2, 3, 4}));
    }

    private static Media uriImage() {
        return new Media(MimeTypeUtils.IMAGE_JPEG, URI.create("https://example.com/cat.jpg"));
    }

    private static Media pdf() {
        return new Media(MimeType.valueOf("application/pdf"), new ByteArrayResource(new byte[] {9}));
    }

    // ── toUrl ────────────────────────────────────────────────────────────

    @Test
    void toUrl_byteBackedMedia_producesBase64DataUrl() {
        assertThat(MediaContentSupport.toUrl(byteImage())).isEqualTo("data:image/jpeg;base64,AQIDBA==");
    }

    @Test
    void toUrl_uriBackedMedia_returnsUriVerbatim() {
        assertThat(MediaContentSupport.toUrl(uriImage())).isEqualTo("https://example.com/cat.jpg");
    }

    @Test
    void toUrl_nullMedia_returnsNull() {
        assertThat(MediaContentSupport.toUrl(null)).isNull();
    }

    // ── imageMediaOf ─────────────────────────────────────────────────────

    @Test
    void imageMediaOf_userMessageWithImage_returnsImage() {
        Message m = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(byteImage()))
                .build();
        assertThat(MediaContentSupport.imageMediaOf(m)).hasSize(1);
    }

    @Test
    void imageMediaOf_filtersOutNonImageMedia() {
        Message m = UserMessage.builder().text("hi").media(List.of(pdf())).build();
        assertThat(MediaContentSupport.imageMediaOf(m)).isEmpty();
    }

    @Test
    void imageMediaOf_messageWithoutMediaSupport_returnsEmpty() {
        assertThat(MediaContentSupport.imageMediaOf(new SystemMessage("you are a bot")))
                .isEmpty();
    }

    @Test
    void imageMediaOf_nullMessage_returnsEmpty() {
        assertThat(MediaContentSupport.imageMediaOf(null)).isEmpty();
    }

    // ── applyImagePolicy: hide flags ─────────────────────────────────────

    @Test
    void applyImagePolicy_defaultConfig_passesUrlThrough() {
        String url = "data:image/jpeg;base64,AQIDBA==";
        assertThat(MediaContentSupport.applyImagePolicy(url, TraceConfig.getDefault(), true))
                .isEqualTo(url);
    }

    @Test
    void applyImagePolicy_hideInputImages_dropsInputUrlOnly() {
        TraceConfig config = TraceConfig.builder().hideInputImages(true).build();
        String url = "data:image/jpeg;base64,AQIDBA==";
        assertThat(MediaContentSupport.applyImagePolicy(url, config, true)).isNull();
        assertThat(MediaContentSupport.applyImagePolicy(url, config, false)).isEqualTo(url);
    }

    @Test
    void applyImagePolicy_hideOutputImages_dropsOutputUrlOnly() {
        TraceConfig config = TraceConfig.builder().hideOutputImages(true).build();
        String url = "data:image/jpeg;base64,AQIDBA==";
        assertThat(MediaContentSupport.applyImagePolicy(url, config, false)).isNull();
        assertThat(MediaContentSupport.applyImagePolicy(url, config, true)).isEqualTo(url);
    }

    // ── applyImagePolicy: base64ImageMaxLength ───────────────────────────

    @Test
    void applyImagePolicy_base64OverMaxLength_redacts() {
        TraceConfig config = TraceConfig.builder().base64ImageMaxLength("10").build();
        assertThat(MediaContentSupport.applyImagePolicy("data:image/jpeg;base64,AQIDBA==", config, true))
                .isEqualTo(MediaContentSupport.REDACTED_VALUE);
    }

    @Test
    void applyImagePolicy_base64UnderMaxLength_passesThrough() {
        TraceConfig config = TraceConfig.builder().base64ImageMaxLength("10000").build();
        String url = "data:image/jpeg;base64,AQIDBA==";
        assertThat(MediaContentSupport.applyImagePolicy(url, config, true)).isEqualTo(url);
    }

    @Test
    void applyImagePolicy_maxLengthDoesNotApplyToHttpUrls() {
        TraceConfig config = TraceConfig.builder().base64ImageMaxLength("5").build();
        String url = "https://example.com/cat.jpg";
        assertThat(MediaContentSupport.applyImagePolicy(url, config, true)).isEqualTo(url);
    }

    @Test
    void applyImagePolicy_unlimitedDefault_neverRedacts() {
        String url = "data:image/jpeg;base64," + "A".repeat(100_000);
        assertThat(MediaContentSupport.applyImagePolicy(url, TraceConfig.getDefault(), true))
                .isEqualTo(url);
    }

    @Test
    void applyImagePolicy_unparseableMaxLength_treatedAsUnlimited() {
        TraceConfig config =
                TraceConfig.builder().base64ImageMaxLength("not-a-number").build();
        String url = "data:image/jpeg;base64,AQIDBA==";
        assertThat(MediaContentSupport.applyImagePolicy(url, config, true)).isEqualTo(url);
    }

    @Test
    void applyImagePolicy_nullUrl_returnsNull() {
        assertThat(MediaContentSupport.applyImagePolicy(null, TraceConfig.getDefault(), true))
                .isNull();
    }
}
