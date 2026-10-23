using NUnit.Framework;
using RepairService.Validators;

[TestFixture]
public class DescriptionValidatorTests
{
    private DescriptionValidator _validator;

    [SetUp]
    public void Setup()
    {
        _validator = new DescriptionValidator();
    }

    [Test]
    public void Validate_DescriptionIsTooShort_ReturnsFalse()
    {
        var result = _validator.Validate("short");
        Assert.IsFalse(result);
    }

    [Test]
    public void Validate_DescriptionIsValid_ReturnsTrue()
    {
        var result = _validator.Validate("Телефон не включается после падения");
        Assert.IsTrue(result);
    }

    [Test]
    public void Validate_DescriptionIsNull_ReturnsFalse()
    {
        var result = _validator.Validate(null);
        Assert.IsFalse(result);
    }
}
