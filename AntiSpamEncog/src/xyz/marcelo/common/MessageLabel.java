package xyz.marcelo.common;

public enum MessageLabel
{
    /** Represents a ham. */
    HAM(new Double[] { 1.0, 0.0 }),

    /** Represents a spam. */
    SPAM(new Double[] { 0.0, 1.0 });

    /** The neural representation of the message. */
    private final Double[] value;

    MessageLabel(Double[] value)
    {
        this.value = value;
    }

    public Double[] getValue()
    {
        return value;
    }
}
