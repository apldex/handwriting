package alpha.AI;

/**
 * Created by apldex on 9/4/16.
 */
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageRecordReader extends BaseImageRecordReader {
    public ImageRecordReader() {
    }

    public ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator) {
        super(height, width, channels, labelGenerator);
    }

    public ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator, double normalizeValue) {
        super(height, width, channels, labelGenerator, (ImageTransform)null, normalizeValue);
    }

    public ImageRecordReader(int height, int width, int channels) {
        super(height, width, channels, (PathLabelGenerator)null);
    }

    public ImageRecordReader(int height, int width, int channels, double normalizeValue) {
        super(height, width, channels, (PathLabelGenerator)null, (ImageTransform)null, normalizeValue);
    }

    public ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator, ImageTransform imageTransform) {
        super(height, width, channels, labelGenerator, imageTransform, 0.0D);
    }

    public ImageRecordReader(int height, int width, int channels, ImageTransform imageTransform) {
        super(height, width, channels, (PathLabelGenerator)null, imageTransform, 0.0D);
    }

    public ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator, ImageTransform imageTransform, double normalizeValue) {
        super(height, width, channels, labelGenerator, imageTransform, normalizeValue);
    }

    public ImageRecordReader(int height, int width, PathLabelGenerator labelGenerator) {
        super(height, width, 1, labelGenerator);
    }

    public ImageRecordReader(int height, int width) {
        super(height, width, 1, (PathLabelGenerator)null, (ImageTransform)null, 0.0D);
    }

    public List<Writable> next() {
        if(this.iter != null) {
            Object ret = new ArrayList();
            File image = (File)this.iter.next();
            this.currentFile = image;
            if(image.isDirectory()) {
                return this.next();
            } else {
                try {
                    this.invokeListeners(image);
                    INDArray e = this.imageLoader.asRowVector(image);
                    ret = RecordConverter.toRecord(e);
                    if(this.appendLabel) {
                        ((List)ret).add(new IntWritable(this.labels.indexOf(this.getLabel(image.getPath()))));
                    }
                } catch (Exception var4) {
                    var4.printStackTrace();
                }

                return (List)ret;
            }
        } else if(this.record != null) {
            this.hitImage = true;
            this.invokeListeners(this.record);
            return this.record;
        } else {
            throw new IllegalStateException("No more elements");
        }
    }
}
