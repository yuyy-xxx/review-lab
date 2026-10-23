using System;

namespace RepairService.Validators
{
    /// <summary>
    /// Валидатор описания заявки на ремонт.
    /// Проверяет длину описания по требованию FR-001 (10-1000 символов).
    /// </summary>
    public class DescriptionValidator
    {
        /// <summary>
        /// Проверяет описание заявки.
        /// </summary>
        /// <param name="description">Описание проблемы</param>
        /// <returns>true если описание корректно, false если слишком короткое</returns>
        public bool Validate(string description)
        {
            if (string.IsNullOrEmpty(description))
            {
                return false;
            }

            if (description.Length < 10)
            {
                return false; // DESCRIPTION_TOO_SHORT
            }

            if (description.Length > 1000)
            {
                return false; // DESCRIPTION_TOO_LONG
            }

            return true;
        }
    }
}
