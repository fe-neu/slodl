#ifndef GRAD_MODE_HPP
#define GRAD_MODE_HPP

/**
 * Whether operations are currently recording for autograd.
 *
 * On by default. While it is off, operations still compute their values but
 * build no graph, so their results are leaves requiring no gradient. This is
 * what makes inference and parameter updates cheap, and what keeps a
 * parameter update from being recorded as part of the model.
 *
 * The flag is per thread: changing it on one thread leaves others alone.
 */
bool is_grad_enabled();

/**
 * Turns recording on or off for the current thread.
 *
 * Prefer NoGradGuard, which cannot forget to restore the previous value.
 *
 * @param enabled  Whether operations should record.
 */
void set_grad_enabled(bool enabled);

/**
 * Switches recording off for as long as this object lives.
 *
 * The constructor stores the current setting and the destructor puts it back,
 * so guards nest correctly and an exception cannot leave recording disabled.
 */
class NoGradGuard {
    public:
        NoGradGuard();
        ~NoGradGuard();

        NoGradGuard(const NoGradGuard&) = delete;
        NoGradGuard& operator=(const NoGradGuard&) = delete;

    private:
        bool previous;
};

#endif
