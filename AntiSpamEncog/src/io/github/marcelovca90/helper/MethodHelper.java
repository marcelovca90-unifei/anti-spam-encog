/**
 *
 */
package io.github.marcelovca90.helper;

import org.apache.commons.lang3.ArrayUtils;

import io.github.marcelovca90.common.MessageLabel;

public class MethodHelper
{
    public static int calculateHiddenLayerSize(int F, int T)
    {
        return (int) Math.ceil((-(F + 2) + Math.sqrt(Math.pow((F + 2), 2) + 4 * T)) / 2);
    }

    public static MessageLabel infer(double[] data)
    {
        if (data.length == 2 && Double.compare(data[0], data[1]) > 0)
            return MessageLabel.HAM;
        else if (data.length == 2 && Double.compare(data[0], data[1]) < 0)
            return MessageLabel.SPAM;
        else
            return null;
    }

    public static MessageLabel infer(Double[] data)
    {
        return infer(ArrayUtils.toPrimitive(data));
    }
}
