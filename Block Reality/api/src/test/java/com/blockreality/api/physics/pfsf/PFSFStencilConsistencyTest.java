package com.blockreality.api.physics.pfsf;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase D — Stencil SSOT 一致性驗證。
 *
 * <p>驗證目標：
 * <ol>
 *   <li>所有 {@code .comp.glsl} shader 都包含 {@code #include "stencil_constants.glsl"}。</li>
 *   <li>所有 {@code .comp.glsl} shader 不再包含 hardcoded {@code const float EDGE_P}
 *       或 {@code const float CORNER_P} 定義（值必須來自 stencil_constants.glsl）。</li>
 *   <li>{@code stencil_constants.glsl} 中的 {@code #define EDGE_P} 與 {@code #define CORNER_P}
 *       數值與 {@link PFSFStencil#EDGE_P} / {@link PFSFStencil#CORNER_P} 一致。</li>
 * </ol>
 *
 * <p>若新增 shader 或修改 {@link PFSFStencil} 常數後，需重新執行：
 * <pre>./gradlew :api:generateStencilGlsl</pre>
 */
class PFSFStencilConsistencyTest {

    private static final String SHADER_RESOURCE_PATH =
        "src/main/resources/assets/blockreality/shaders/compute/pfsf";

    /** 不需要包含 stencil include 的非 compute shader / 工具 glsl */
    private static final java.util.Set<String> INCLUDE_EXEMPTIONS = java.util.Set.of(
        "stencil_constants.glsl",
        "morton_utils.glsl",
        "energy_common.glsl",
        // Label-propagation shaders 做 BFS 連通分量，不使用 26 連通 Laplacian
        // (EDGE_P / CORNER_P)。整條 GPU label-prop 路徑已於 PR2 (audit-fixes)
        // 切離 Java 端 collapse pipeline；shader 本身待後續 PR 一併移除。
        "label_prop_init.comp.glsl",
        "label_prop_iterate.comp.glsl",
        "label_prop_summarise_aggregate.comp.glsl",
        "label_prop_summarise_alloc.comp.glsl"
    );

    @Test
    void allComputeShadersIncludeStencilConstants() throws IOException {
        Path shaderDir = findShaderDir();
        List<String> missing = new ArrayList<>();

        try (Stream<Path> stream = Files.walk(shaderDir)) {
            stream.filter(p -> p.toString().endsWith(".comp.glsl"))
                  .filter(p -> !INCLUDE_EXEMPTIONS.contains(p.getFileName().toString()))
                  .forEach(shader -> {
                      try {
                          String content = Files.readString(shader);
                          if (!content.contains("#include \"stencil_constants.glsl\"")) {
                              missing.add(shader.getFileName().toString());
                          }
                      } catch (IOException e) {
                          missing.add(shader.getFileName() + " (read error: " + e.getMessage() + ")");
                      }
                  });
        }

        assertTrue(missing.isEmpty(),
            "以下 compute shader 缺少 #include \"stencil_constants.glsl\":\n  " +
            String.join("\n  ", missing) + "\n" +
            "請在 #version / #extension 後加入：\n" +
            "  #extension GL_GOOGLE_include_directive : enable\n" +
            "  #include \"stencil_constants.glsl\"");
    }

    @Test
    void noShaderHasHardcodedEdgePOrCornerPConst() throws IOException {
        Path shaderDir = findShaderDir();
        Pattern hardcodedPattern = Pattern.compile(
            "const\\s+float\\s+(EDGE_P|CORNER_P)\\s*=\\s*[\\d.]+"
        );
        List<String> violations = new ArrayList<>();

        try (Stream<Path> stream = Files.walk(shaderDir)) {
            stream.filter(p -> p.toString().endsWith(".comp.glsl"))
                  .filter(p -> !INCLUDE_EXEMPTIONS.contains(p.getFileName().toString()))
                  .forEach(shader -> {
                      try {
                          String content = Files.readString(shader);
                          Matcher m = hardcodedPattern.matcher(content);
                          if (m.find()) {
                              violations.add(shader.getFileName() + ": \"" + m.group() + "\"");
                          }
                      } catch (IOException e) {
                          violations.add(shader.getFileName() + " (read error)");
                      }
                  });
        }

        assertTrue(violations.isEmpty(),
            "以下 shader 仍有 hardcoded EDGE_P / CORNER_P 常數定義（應改用 stencil_constants.glsl）:\n  " +
            String.join("\n  ", violations));
    }

    @Test
    void stencilConstantsGlslMatchesPFSFStencilJava() throws IOException {
        Path glslFile = findShaderDir().resolve("stencil_constants.glsl");
        assertTrue(Files.exists(glslFile),
            "stencil_constants.glsl 不存在，請執行: ./gradlew :api:generateStencilGlsl");

        String content = Files.readString(glslFile);

        float expectedEdgeP   = PFSFStencil.EDGE_P;
        float expectedCornerP = PFSFStencil.CORNER_P;

        // 解析 #define EDGE_P <value>
        Pattern edgePattern   = Pattern.compile("#define\\s+EDGE_P\\s+([\\d.]+)");
        Pattern cornerPattern = Pattern.compile("#define\\s+CORNER_P\\s+([\\d.]+)");

        Matcher em = edgePattern.matcher(content);
        assertTrue(em.find(), "stencil_constants.glsl 缺少 #define EDGE_P");
        float glslEdgeP = Float.parseFloat(em.group(1));

        Matcher cm = cornerPattern.matcher(content);
        assertTrue(cm.find(), "stencil_constants.glsl 缺少 #define CORNER_P");
        float glslCornerP = Float.parseFloat(cm.group(1));

        assertEquals(expectedEdgeP, glslEdgeP, 1e-6f,
            "EDGE_P 不一致：PFSFStencil.EDGE_P=" + expectedEdgeP +
            " 但 stencil_constants.glsl 中為 " + glslEdgeP +
            "。請執行: ./gradlew :api:generateStencilGlsl");

        assertEquals(expectedCornerP, glslCornerP, 1e-6f,
            "CORNER_P 不一致：PFSFStencil.CORNER_P=" + expectedCornerP +
            " 但 stencil_constants.glsl 中為 " + glslCornerP +
            "。請執行: ./gradlew :api:generateStencilGlsl");
    }

    @Test
    void pfsfConstantsDelegatesAreConsistent() {
        assertEquals(PFSFStencil.EDGE_P,   PFSFConstants.SHEAR_EDGE_PENALTY,   1e-6f,
            "PFSFConstants.SHEAR_EDGE_PENALTY 應等於 PFSFStencil.EDGE_P");
        assertEquals(PFSFStencil.CORNER_P, PFSFConstants.SHEAR_CORNER_PENALTY, 1e-6f,
            "PFSFConstants.SHEAR_CORNER_PENALTY 應等於 PFSFStencil.CORNER_P");
    }

    /**
     * Phase D retrofit：驗證 INJECT fallback 模式的 regex 展開結果與 INCLUDE 等價。
     *
     * <p>模擬 {@code generateStencilGlsl -Pstencil.mode=INJECT} 的 regex 展開邏輯：
     * 對每個 shader 將 {@code #include "stencil_constants.glsl"} 與 GOOGLE include
     * 擴充指令一併替換為內聯 {@code #define} block。驗證：
     * <ol>
     *   <li>每個 shader 在展開後都能找到 EDGE_P 與 CORNER_P 的 {@code #define}</li>
     *   <li>內聯的 {@code #define} 數值與 {@link PFSFStencil} 常數一致</li>
     *   <li>展開後 shader 不再有 unresolved {@code #include} 指令</li>
     * </ol>
     */
    @Test
    void injectModeProducesEquivalentExpansion() throws IOException {
        Path shaderDir = findShaderDir();
        String edgeP   = String.valueOf(PFSFStencil.EDGE_P);
        String cornerP = String.valueOf(PFSFStencil.CORNER_P);
        String inlineBlock =
            "// ─── BEGIN inlined from stencil_constants.glsl (INJECT mode) ───\n" +
            "#ifndef STENCIL_CONSTANTS_GLSL\n" +
            "#define STENCIL_CONSTANTS_GLSL\n" +
            "#define EDGE_P   " + edgeP + "\n" +
            "#define CORNER_P " + cornerP + "\n" +
            "#endif\n" +
            "// ─── END inlined from stencil_constants.glsl ───";

        Pattern includePattern =
            Pattern.compile("#extension\\s+GL_GOOGLE_include_directive\\s*:\\s*enable\\s*\\n?");
        Pattern directivePattern =
            Pattern.compile("#include\\s+\"stencil_constants\\.glsl\"\\s*\\n?");
        Pattern definePattern =
            Pattern.compile("#define\\s+(EDGE_P|CORNER_P)\\s+([\\d.]+)");

        List<String> failures = new ArrayList<>();

        try (Stream<Path> stream = Files.walk(shaderDir)) {
            stream.filter(p -> p.toString().endsWith(".comp.glsl"))
                  .filter(p -> !INCLUDE_EXEMPTIONS.contains(p.getFileName().toString()))
                  .forEach(shader -> {
                      try {
                          String original = Files.readString(shader);
                          if (!original.contains("#include \"stencil_constants.glsl\"")) {
                              // Phase D 測試前面已驗證 include 存在，這邊只是雙保險
                              return;
                          }
                          String expanded = directivePattern.matcher(
                              includePattern.matcher(original).replaceFirst("")
                          ).replaceFirst(Matcher.quoteReplacement(inlineBlock + "\n"));

                          // 1. 展開後不應再有 #include
                          if (expanded.contains("#include \"stencil_constants.glsl\"")) {
                              failures.add(shader.getFileName() + ": #include still present after INJECT");
                              return;
                          }

                          // 2. 必須能找到 EDGE_P 與 CORNER_P 的 #define
                          Matcher defs = definePattern.matcher(expanded);
                          boolean edgeFound = false, cornerFound = false;
                          while (defs.find()) {
                              String name = defs.group(1);
                              float value = Float.parseFloat(defs.group(2));
                              if ("EDGE_P".equals(name)) {
                                  edgeFound = true;
                                  if (Math.abs(value - PFSFStencil.EDGE_P) > 1e-6f) {
                                      failures.add(shader.getFileName() +
                                          ": injected EDGE_P=" + value +
                                          " mismatches PFSFStencil.EDGE_P=" + PFSFStencil.EDGE_P);
                                  }
                              } else if ("CORNER_P".equals(name)) {
                                  cornerFound = true;
                                  if (Math.abs(value - PFSFStencil.CORNER_P) > 1e-6f) {
                                      failures.add(shader.getFileName() +
                                          ": injected CORNER_P=" + value +
                                          " mismatches PFSFStencil.CORNER_P=" + PFSFStencil.CORNER_P);
                                  }
                              }
                          }
                          if (!edgeFound) failures.add(shader.getFileName() + ": INJECT did not yield #define EDGE_P");
                          if (!cornerFound) failures.add(shader.getFileName() + ": INJECT did not yield #define CORNER_P");
                      } catch (IOException e) {
                          failures.add(shader.getFileName() + " (read error)");
                      }
                  });
        }

        assertTrue(failures.isEmpty(),
            "INJECT 模式展開與 PFSFStencil SSOT 不一致:\n  " +
            String.join("\n  ", failures));
    }

    private Path findShaderDir() {
        // 從不同 working directory 都能找到 shader 目錄（IDE / Gradle 執行路徑不同）
        String[] candidates = {
            SHADER_RESOURCE_PATH,
            "api/" + SHADER_RESOURCE_PATH,
            "../api/" + SHADER_RESOURCE_PATH,
        };
        for (String candidate : candidates) {
            Path p = Paths.get(candidate);
            if (Files.isDirectory(p)) return p;
        }
        // Gradle test 工作目錄通常在 api/
        Path fromProject = Paths.get(System.getProperty("user.dir"), SHADER_RESOURCE_PATH);
        if (Files.isDirectory(fromProject)) return fromProject;

        throw new RuntimeException(
            "找不到 shader 目錄，請確認從 'Block Reality/api/' 或 'Block Reality/' 下執行 Gradle"
        );
    }
}
