package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;

class SpringAIInstrumentorMediaTest {

    private static final String JPEG_DATA_URL = "data:image/jpeg;base64,AQIDBA==";

    private InMemorySpanExporter spanExporter;
    private SdkTracerProvider tracerProvider;

    @BeforeEach
    void setUp() {
        spanExporter = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
                .build();
    }

    @AfterEach
    void tearDown() {
        spanExporter.reset();
        tracerProvider.close();
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private static Media jpeg() {
        return new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(new byte[] {1, 2, 3, 4}));
    }

    private SpringAIInstrumentor instrumentor(TraceConfig config) {
        return new SpringAIInstrumentor(new OITracer(tracerProvider.get("test"), config));
    }

    private ChatModelObservationContext context(List<Message> inputs, AssistantMessage output) {
        Prompt prompt = mock(Prompt.class);
        when(prompt.getInstructions()).thenReturn(inputs);
        ChatOptions options = mock(ChatOptions.class);
        when(options.getModel()).thenReturn("gpt-4o");
        when(prompt.getOptions()).thenReturn(options);

        ChatModelObservationContext ctx = mock(ChatModelObservationContext.class);
        when(ctx.getRequest()).thenReturn(prompt);
        if (output != null) {
            ChatResponse response = mock(ChatResponse.class);
            when(response.getResults()).thenReturn(List.of(new Generation(output)));
            when(ctx.getResponse()).thenReturn(response);
        }
        return ctx;
    }

    private SpanData run(SpringAIInstrumentor instrumentor, ChatModelObservationContext ctx) {
        instrumentor.onStart(ctx);
        instrumentor.onStop(ctx);
        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        return spans.get(0);
    }

    private static String attr(SpanData span, String key) {
        return span.getAttributes().get(AttributeKey.stringKey(key));
    }

    // ── Input side ──────────────────────────────────────────────────────

    @Test
    void userMessageWithImage_emitsMultiPartContents() {
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.role")).isEqualTo("user");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Describe this image.");
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.type"))
                .isEqualTo("image");
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);

        // Multi-part replaces the flat content attribute.
        assertThat(attr(span, "llm.input_messages.0.message.content")).isNull();
    }

    @Test
    void userMessageWithoutMedia_keepsFlatContent() {
        SpanData span =
                run(instrumentor(TraceConfig.getDefault()), context(List.of(new UserMessage("Hello world")), null));

        assertThat(attr(span, "llm.input_messages.0.message.content")).isEqualTo("Hello world");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isNull();
    }

    @Test
    void nonImageMedia_fallsBackToFlatContent() {
        Media pdf = new Media(MimeType.valueOf("application/pdf"), new ByteArrayResource(new byte[] {9}));
        Message message =
                UserMessage.builder().text("Read this").media(List.of(pdf)).build();
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.content")).isEqualTo("Read this");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isNull();
    }

    @Test
    void imageWithoutText_startsContentsAtIndexZero() {
        Message message = UserMessage.builder().text("").media(List.of(jpeg())).build();
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("image");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
    }

    @Test
    void multipleImages_areIndexedSequentially() {
        Message message = UserMessage.builder()
                .text("Compare these")
                .media(List.of(jpeg(), jpeg()))
                .build();
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
        assertThat(attr(span, "llm.input_messages.0.message.contents.2.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
    }

    @Test
    void inputValueJson_containsMultiPartContent() {
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(message), null));

        String inputValue = attr(span, SemanticConventions.INPUT_VALUE);
        assertThat(inputValue).contains("\"type\":\"text\"").contains("\"type\":\"image_url\"");
        assertThat(inputValue).contains(JPEG_DATA_URL);
    }

    // ── Output side ─────────────────────────────────────────────────────

    @Test
    void assistantMessageWithImage_emitsMultiPartContents() {
        AssistantMessage output = new AssistantMessage("Here it is", java.util.Map.of(), List.of(), List.of(jpeg()));
        SpanData span =
                run(instrumentor(TraceConfig.getDefault()), context(List.of(new UserMessage("draw a cat")), output));

        assertThat(attr(span, "llm.output_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(attr(span, "llm.output_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
        assertThat(attr(span, "llm.output_messages.0.message.content")).isNull();
    }

    // ── Privacy and size policy ─────────────────────────────────────────

    @Test
    void hideInputImages_dropsUrlButKeepsPartMarker() {
        TraceConfig config = TraceConfig.builder().hideInputImages(true).build();
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(config), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.type"))
                .isEqualTo("image");
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isNull();
        // The text part is unaffected.
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Describe this image.");
        // And the base64 must not leak through input.value either.
        assertThat(attr(span, SemanticConventions.INPUT_VALUE)).doesNotContain(JPEG_DATA_URL);
    }

    @Test
    void hideOutputImages_doesNotAffectInputImages() {
        TraceConfig config = TraceConfig.builder().hideOutputImages(true).build();
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(config), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
    }

    @Test
    void base64ImageMaxLength_redactsOversizedImage() {
        TraceConfig config = TraceConfig.builder().base64ImageMaxLength("10").build();
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(config), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo("__REDACTED__");
        assertThat(attr(span, SemanticConventions.INPUT_VALUE)).doesNotContain(JPEG_DATA_URL);
    }

    @Test
    void hideInputMessages_stillSuppressesMultiPartContents() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        Message message = UserMessage.builder()
                .text("Describe this image.")
                .media(List.of(jpeg()))
                .build();
        SpanData span = run(instrumentor(config), context(List.of(message), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isNull();
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isNull();
    }
}
