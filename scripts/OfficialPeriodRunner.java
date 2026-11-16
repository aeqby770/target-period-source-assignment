import ed.biodare.period.mesa.MESAAnalyser;
import ed.biodare.period.mfourfit.MFFAnalyser;
import ed.robust.dom.data.TimeSeries;
import ed.robust.dom.tsprocessing.PPAResult;
import ed.robust.util.timeseries.TimeSeriesOperations;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class OfficialPeriodRunner {
    public static void main(String[] args) throws Exception {
        Path input = Path.of(args[0]);
        Path output = Path.of(args[1]);
        double periodMin = Double.parseDouble(args[2]);
        double periodMax = Double.parseDouble(args[3]);
        int maxTraces = Integer.parseInt(args[4]);
        String detrending = args[5];
        Map<String, TimeSeries> traces = new LinkedHashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split("\t", 3);
                traces.computeIfAbsent(fields[0], key -> new TimeSeries()).add(
                    Double.parseDouble(fields[1]), Double.parseDouble(fields[2])
                );
            }
        }
        List<Map.Entry<String, TimeSeries>> entries = new ArrayList<>(traces.entrySet());
        int count = maxTraces > 0 ? Math.min(maxTraces, entries.size()) : entries.size();
        String[] rows = new String[count];
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("trace_id,method,period_h,gof,method_error,status\n");
            for (int start = 0; start < count; start += 50) {
                int end = Math.min(start + 50, count);
                IntStream.range(start, end).parallel().forEach(index -> {
                    Map.Entry<String, TimeSeries> entry = entries.get(index);
                    TimeSeries series = entry.getValue();
                    if (detrending.equals("linear")) {
                        series = TimeSeriesOperations.lineDetrended(series);
                    }
                    String mesa;
                    String mfourfit;
                    try {
                        PPAResult result = new MESAAnalyser().analyse(series, periodMin, periodMax);
                        mesa = entry.getKey() + ",MESA," + result.getPeriod() + ","
                            + result.getPPAMethodSpecific().getGOF() + ","
                            + result.getPPAMethodSpecific().getMethodError() + ",ok\n";
                    } catch (Exception error) {
                        mesa = entry.getKey() + ",MESA,NaN,NaN,NaN,"
                            + error.getClass().getSimpleName() + "\n";
                    }
                    try {
                        PPAResult result = new MFFAnalyser().analyse(series, periodMin, periodMax);
                        mfourfit = entry.getKey() + ",MFourFit," + result.getPeriod() + ","
                            + result.getPPAMethodSpecific().getGOF() + ","
                            + result.getPPAMethodSpecific().getMethodError() + ",ok\n";
                    } catch (Exception error) {
                        mfourfit = entry.getKey() + ",MFourFit,NaN,NaN,NaN,"
                            + error.getClass().getSimpleName() + "\n";
                    }
                    rows[index] = mesa + mfourfit;
                });
                for (int index = start; index < end; index++) {
                    writer.write(rows[index]);
                }
                writer.flush();
                System.err.println("completed=" + end);
            }
        }
    }
}
